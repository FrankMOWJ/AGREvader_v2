import numpy as np
from constants import *
from sklearn.cluster import KMeans


class Aggregator:
    """
    The aggregator class collecting gradients calculated by participants and plus together
    """

    def __init__(self, sample_gradients: torch.Tensor, robust_mechanism=None):
        """
        Initiate the aggregator according to the tensor size of a given sample
        :param sample_gradients: The tensor size of a sample gradient
        :param robust_mechanism: the robust mechanism used to aggregate the gradients
        """
        self.sample_gradients = sample_gradients.to(DEVICE)
        self.collected_gradients = []
        self.counter = 0
        self.counter_by_indices = torch.ones(self.sample_gradients.size()).to(DEVICE)
        self.robust = RobustMechanism(robust_mechanism)

        # AGR related parameters
        self.agr_model = None #Global model Fang required

    def reset(self):
        """
        Reset the aggregator to 0 before next round of aggregation
        """
        self.collected_gradients = []
        self.counter = 0
        self.counter_by_indices = torch.ones(self.sample_gradients.size()).to(DEVICE)
        self.agr_model_calculated = False

    def collect(self, gradient: torch.Tensor,source, indices=None, sample_count=None):
        """
        Collect one set of gradients from a participant
        :param gradient: The gradient calculated by a participant
        :param souce: The source of the gradient, used for AGR verification
        :param indices: the indices of the gradient, used for AGR verification

        """
        if sample_count is None:
            self.collected_gradients.append(gradient)
            if indices is not None:
                # print(indices.device)
                # print(self.counter_by_indices.device)
                self.counter_by_indices[indices] += 1
            self.counter += 1
        else:
            self.collected_gradients.append(gradient * sample_count)
            if indices is not None:
                self.counter_by_indices[indices] += sample_count
            self.counter += sample_count

    def get_outcome(self, reset=False, by_indices=False):
        """
        Get the aggregated gradients and reset the aggregator if needed, apply robust aggregator mechanism if needed
        :param reset: Whether to reset the aggregator after getting the outcome
        :param by_indices: Whether to aggregate by indices
        """
        if by_indices:
            result = sum(self.collected_gradients) / self.counter_by_indices
        else:
            result = self.robust.getter(self.collected_gradients, malicious_user=NUMBER_OF_ADVERSARY)
        if reset:
            self.reset()
        return result

    def agr_model_acquire(self, model):
        """
        Make use of the given model for AGR verification
        :param model: The model used for AGR verification
        """
        self.agr_model = model
        self.robust.agr_model_acquire(model)



class RobustMechanism:
    """
    The robust aggregator applied in the aggregator
    """
    #predefined the list of participants indices and status in AGR
    appearence_list = [0,1,2,3,4,5]
    status_list = []

    def __init__(self, robust_mechanism):
        self.type = robust_mechanism
        if robust_mechanism == NONE:
            self.function = self.naive_average
        elif robust_mechanism == MEDIAN:
            self.function = self.median
        elif robust_mechanism == FANG:
            self.function = self.Fang_defense
        elif robust_mechanism == TRIM:
            self.function = self.trimmed_mean
        elif robust_mechanism == KRUM:
            self.function = self.krum
        elif robust_mechanism == MULTI_KRUM:
            self.function = self.multi_krum
        elif robust_mechanism == DEEPSIGHT:
            self.function = self.deepsight
        self.agr_model = None


    def agr_model_acquire(self, model: torch.nn.Module):
        """
        Acquire the model used for LRR and ERR verification in Fang Defense
        The model must have the same parameters as the global model
        :param model: The model used for LRR and ERR verification
        """
        self.agr_model = model

    def naive_average(self, input_gradients: torch.Tensor):
        """
        The naive aggregator
        :param input_gradients: The gradients collected from participants
        :return: The average of the gradients
        """
        return torch.mean(input_gradients, 0)

    def median(self, input_gradients: torch.Tensor,number_of_malicious):
        """
        The median AGR
        :param input_gradients: The gradients collected from participants
        :return: The median of the gradients
        """
        return torch.median(input_gradients, 0).values
    
    def trimmed_mean(self, input_gradients: torch.Tensor, malicious_user: int, trim_bound=TRIM_BOUND):
        """
        The trimmed mean AGR
        :param input_gradients: The gradients collected from participants
        :param malicious_user: The number of malicious participants
        :param trim_bound: The number of gradients to be trimmed from both ends
        :return: The trimmed mean of the gradients
        """
        input_gradients = torch.sort(input_gradients, dim=0).values
        print("ok")
        return input_gradients[trim_bound:-trim_bound].mean(0)
        
    def krum(self, input_gradients: torch.Tensor, malicious_user: int):
        '''
        The Krum mechanism
        :param input_gradients: The gradients collected from participants
        :param malicious_user: The number of malicious participants
        :return: The average of the gradients after removing the malicious participants
        '''       
        num_participants = input_gradients.shape[0]
        num_select = num_participants - malicious_user - 2
        
        if num_select <= 0:
            raise ValueError("The number of malicious users is too high, making selection impossible.")
        
        # Step 1: Calculate the distances between each pair of gradients
        distances = torch.zeros((num_participants, num_participants))
        for i in range(num_participants):
            for j in range(i + 1, num_participants):
                distances[i, j] = torch.norm(input_gradients[i] - input_gradients[j])
                distances[j, i] = distances[i, j]

        # Step 2: For each gradient, calculate the sum of distances to the closest num_select gradients
        scores = []
        for i in range(num_participants):
            sorted_distances, _ = torch.sort(distances[i])
            score = torch.sum(sorted_distances[:num_select])
            scores.append(score)
        
        # Step 3: Select the gradient with the smallest score
        selected_index = torch.argmin(torch.tensor(scores))
        selected_gradient = input_gradients[selected_index]
        
        return selected_gradient
    
    def multi_krum(self, input_gradients: torch.Tensor, malicious_user: int, num_select=2):
        '''
        The Multi-Krum mechanism
        :param input_gradients: The gradients collected from participants
        :param malicious_user: The number of malicious participants
        :param num_select: The number of gradients to be selected
        :return: The average of the gradients after removing the malicious participants
        '''       
        num_participants = input_gradients.shape[0]
        num_to_select = num_participants - malicious_user - 2

        # Check if num_to_select is valid
        if num_to_select <= 0:
            raise ValueError("The number of malicious users is too high, making selection impossible.")

        # Step 1: Calculate the distances between each pair of gradients
        distances = torch.zeros((num_participants, num_participants))
        for i in range(num_participants):
            for j in range(i + 1, num_participants):
                distances[i, j] = torch.norm(input_gradients[i] - input_gradients[j])
                distances[j, i] = distances[i, j]

        # Step 2: For each gradient, calculate the sum of distances to the closest num_to_select gradients
        scores = []
        for i in range(num_participants):
            sorted_distances, _ = torch.sort(distances[i])
            score = torch.sum(sorted_distances[:num_to_select])
            scores.append(score)

        # Step 3: Select the num_select gradients with the smallest scores
        selected_indices = torch.argsort(torch.tensor(scores))[:num_select]
        selected_gradients = input_gradients[selected_indices]

        # Step 4: Return the average of the selected gradients
        global_gradient = torch.mean(selected_gradients, dim=0)
        
        # print("********************************")
        # print("Selected indices: ", selected_indices)
        # print("********************************")
        
        return global_gradient
    
    def Fang_defense(self, input_gradients: torch.Tensor, malicious_user: int):
        """
        The LRR and ERR mechanism proposed in Fang defense
        :param input_gradients: The gradients collected from participants
        :param malicious_user: The number of malicious participants
        :return: The average of the gradients after removing the malicious participants
        """
        # Get the baseline loss and accuracy without removing any of the inputs
        all_avg = torch.mean(input_gradients, 0)
        base_loss, base_acc = self.agr_model.test_gradients(all_avg)
        loss_impact = []
        err_impact = []
        # Get all the loss value and accuracy without ith input
        RobustMechanism.status_list = []
        for i in range(len(input_gradients)):
            avg_without_i = (sum(input_gradients[:i]) + sum(input_gradients[i+1:])) / (input_gradients.size(0) - 1)
            ith_loss, ith_acc = self.agr_model.test_gradients(avg_without_i)
            loss_impact.append(torch.tensor(base_loss - ith_loss))
            err_impact.append(torch.tensor(ith_acc - base_acc))
            RobustMechanism.status_list.append((i,ith_acc,ith_loss))
        loss_impact = torch.hstack(loss_impact)
        err_impact = torch.hstack(err_impact)
        loss_rank = torch.argsort(loss_impact, dim=-1)
        acc_rank = torch.argsort(err_impact, dim=-1)
        result = []
        for i in range(len(input_gradients)):
            if i in loss_rank[:-malicious_user] and i in acc_rank[:-malicious_user]:
                result.append(i)
        RobustMechanism.appearence_list = result
        return torch.mean(input_gradients[result], dim=0)

    def deepsight(self, input_gradients: torch.Tensor, malicious_user: int, num_clusters=2):
        '''
        The Deepsight mechanism
        :param input_gradients: The gradients collected from participants (shape: [num_users, num_features])
        :param malicious_user: The number of malicious participants
        :param num_clusters: The number of clusters for KMeans
        :return: The aggregated global gradient
        '''

        # Compute DDifs and NEUPs
        DDifs = torch.norm(torch.mean(input_gradients, dim=0, keepdim=True) - input_gradients, p=2, dim=1)
        NEUPs = torch.norm(input_gradients, p=2, dim=1)

        # Stack features for clustering
        features = torch.stack([DDifs, NEUPs], dim=1).cpu().numpy()

        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(features)
        labels = kmeans.labels_

        # Identify clusters for benign and poisoned updates
        benign_cluster = np.argmax(np.bincount(labels))
        poisoned_cluster = 1 - benign_cluster
        
        # Mark the gradients based on clusters
        benign_gradients = input_gradients[labels == benign_cluster]
        poisoned_gradients = input_gradients[labels == poisoned_cluster]
        
        max_norm = 1.0
        # Apply ℓ2 norm clipping
        def clip_gradients(gradients):
            norms = torch.norm(gradients, p=2, dim=1)
            clip_factor = max_norm / (norms + 1e-8)  # Avoid division by zero
            clip_factor = torch.clamp(clip_factor, max=1.0)
            return gradients * clip_factor.unsqueeze(1)

        benign_gradients = clip_gradients(benign_gradients)

        # Aggregate benign gradients
        if benign_gradients.size(0) > 0:
            aggregated_gradient = torch.mean(benign_gradients, dim=0)
        else:
            aggregated_gradient = torch.mean(input_gradients, dim=0)  # Fall back to mean of all gradients if no benign updates

        return aggregated_gradient
    
    def RFLBAT(self, input_gradients: torch.Tensor, malicious_user: int):
        '''
        The RFLBAT mechanism
        :param input_gradients: The gradients collected from participants
        :param malicious_user: The number of malicious participants
        :return: The average of the gradients after removing the malicious participants
        '''
        
        pass
    
    def FLAME(self, input_gradients: torch.Tensor, malicious_user: int):
        '''
        The FLAME mechanism
        :param input_gradients: The gradients collected from participants
        :param malicious_user: The number of malicious participants
        :return: The average of the gradients after removing the malicious participants
        '''
        
        pass
    
        
        
    def getter(self, gradients, malicious_user=NUMBER_OF_ADVERSARY):
        """
        The getter method applying the robust AGR
        :param gradients: The gradients collected from all participants
        :param malicious_user: The number of malicious participants
        :return: The average of the gradients after adding the malicious gradient
        """
        gradients = torch.vstack(gradients)
        if self.function == self.naive_average or self.function == self.median:
            return self.function(gradients)
        elif self.function == self.trimmed_mean:
            return self.function(gradients, malicious_user, TRIM_BOUND)
        elif self.function == self.krum or self.function == self.Fang_defense:
            return self.function(gradients, malicious_user)
        elif self.function == self.multi_krum:
            return self.function(gradients, malicious_user, MULTI_KRUM_K)
        elif self.function == self.deepsight:
            return self.function(gradients, malicious_user, NUM_CLUSTER)
        else:
            raise ValueError("The robust mechanism is not implemented yet.")