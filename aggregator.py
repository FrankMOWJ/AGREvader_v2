import numpy as np
from constants import *
from sklearn.cluster import KMeans
import torch.nn.functional as F


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
        elif robust_mechanism == RFLBAT:
            self.function = self.RFLBAT
        elif robust_mechanism == FLAME:
            self.function = self.FLAME
        elif robust_mechanism == FOOLSGOLD:
            self.function = self.Foolsgold
        self.agr_model = None
        self.history_gradients = None  # To store historical gradients for each client


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

    def median(self, input_gradients: torch.Tensor):
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
    
    def RFLBAT(self, input_gradients: torch.Tensor, malicious_user: int, num_components=2):
        '''
        The RFLBAT mechanism
        :param input_gradients: The gradients collected from participants
        :param malicious_user: The number of malicious participants
        :return: The average of the gradients after removing the malicious participants
        '''
        
        # Step 1: PCA for dimensionality reduction using torch.pca_lowrank
        U, S, V = torch.pca_lowrank(input_gradients, q=num_components)
        X_dr = torch.mm(input_gradients, V[:, :num_components])
        
        # Step 2: Pair-wise Euclidean distance and outlier filtering
        eu_list = []
        for i in range(len(X_dr)):
            eu_sum = 0
            for j in range(len(X_dr)):
                if i == j:
                    continue
                eu_sum += torch.norm(X_dr[i] - X_dr[j]).item()
            eu_list.append(eu_sum)
        
        # Filter outliers based on a threshold
        threshold1 = torch.median(torch.tensor(eu_list)) * 1.5  # 1.5 is an arbitrary factor
        accept_indices = [i for i, eu_sum in enumerate(eu_list) if eu_sum < threshold1]

        # Step 3: K-means clustering using torch
        accepted_X_dr = X_dr[accept_indices]
        kmeans = KMeans(n_clusters=num_components)
        labels = kmeans.fit_predict(accepted_X_dr.cpu().numpy())
        
        # Step 4: Select the optimal cluster based on cosine similarity
        min_cosine_similarity = float('inf')
        optimal_cluster_indices = []
        for i in range(num_components):
            cluster_indices = [idx for idx, label in enumerate(labels) if label == i]
            cluster_data = accepted_X_dr[cluster_indices]
            mean_gradient = torch.mean(cluster_data, dim=0)
            cosine_similarities = F.cosine_similarity(cluster_data, mean_gradient.unsqueeze(0), dim=1)
            avg_cosine_similarity = torch.mean(cosine_similarities)
            if avg_cosine_similarity < min_cosine_similarity:
                min_cosine_similarity = avg_cosine_similarity
                optimal_cluster_indices = cluster_indices
        
        # Step 5: Final outlier filtering in the optimal cluster
        final_accept_indices = []
        optimal_cluster_data = accepted_X_dr[optimal_cluster_indices]
        final_eu_list = []
        for i in range(len(optimal_cluster_data)):
            eu_sum = 0
            for j in range(len(optimal_cluster_data)):
                if i == j:
                    continue
                eu_sum += torch.norm(optimal_cluster_data[i] - optimal_cluster_data[j]).item()
            final_eu_list.append(eu_sum)
        
        threshold2 = torch.median(torch.tensor(final_eu_list)) * 1.5  # 1.5 is an arbitrary factor
        final_accept_indices = [optimal_cluster_indices[i] for i, eu_sum in enumerate(final_eu_list) if eu_sum < threshold2]
        
        # Step 6: Calculate and return the average of the remaining gradients
        accepted_gradients = input_gradients[final_accept_indices]
        average_gradient = torch.mean(accepted_gradients, dim=0)
        
        return average_gradient
    
    def FLAME(self, input_gradients: torch.Tensor, malicious_user: int):
        '''
        The FLAME mechanism
        :param input_gradients: The gradients collected from participants
        :param malicious_user: The number of malicious participants
        :return: The average of the gradients after removing the malicious participants
        '''
        
        # Step 1: Calculate the mean and standard deviation of the gradients
        mean_gradient = torch.mean(input_gradients, dim=0)
        std_gradient = torch.std(input_gradients, dim=0)
        
        # Step 2: Calculate z-scores for each participant's gradients
        z_scores = torch.abs((input_gradients - mean_gradient) / std_gradient)
        
        # Step 3: Identify and remove the gradients of malicious users
        # Here we assume that the malicious users have high z-scores
        threshold = torch.topk(z_scores, malicious_user, dim=0).values[-1]
        mask = torch.all(z_scores <= threshold, dim=1)
        filtered_gradients = input_gradients[mask]
        
        # Step 4: Return the average of the filtered gradients
        return torch.mean(filtered_gradients, dim=0)

    def Foolsgold(self, input_gradients: torch.Tensor, malicious_user: int):
        '''
        The Foolsgold mechanism
        :param input_gradients: The gradients collected from participants, shape (num_participants, num_params)
        :param malicious_user: The number of malicious participants
        :return: The average of the gradients after removing the malicious participants
        '''
        
        # 1. Update historical gradients
        if self.history_gradients is None:
            self.history_gradients = input_gradients.unsqueeze(0)
        else:
            self.history_gradients = torch.cat((self.history_gradients, input_gradients.unsqueeze(0)), dim=0)
        
        num_participants = input_gradients.shape[0]

        # 2. Compute cosine similarity between pair-wise historical updates
        history_cos_sim = F.cosine_similarity(self.history_gradients.unsqueeze(1), self.history_gradients.unsqueeze(2), dim=3).to(DEVICE)

        # 3. Initialize credit scores
        credit_scores = torch.ones(num_participants, device=DEVICE)

        # 4. Adjust credit scores based on cosine similarity
        for i in range(num_participants):
            for j in range(num_participants):
                if i != j:
                    credit_scores[i] *= (1 - history_cos_sim[-1, i, j])  # Use the latest updates' similarity

        # 5. Inverse of credit score to penalize high similarity
        weights = 1.0 / (credit_scores + 1e-6)  # avoid division by zero
        weights = weights / torch.sum(weights)  # normalize weights

        # 6. Compute the weighted average of gradients
        weighted_avg_gradient = torch.matmul(weights.unsqueeze(0), input_gradients).squeeze()

        return weighted_avg_gradient    
        
        
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
        elif self.function == self.krum or self.function == self.FLAME \
            or self.function == self.Foolsgold or self.function == self.Fang_defense:
            return self.function(gradients, malicious_user)
        elif self.function == self.trimmed_mean:
            return self.function(gradients, malicious_user, TRIM_BOUND)
        elif self.function == self.multi_krum:
            return self.function(gradients, malicious_user, MULTI_KRUM_K)
        elif self.function == self.deepsight:
            return self.function(gradients, malicious_user, NUM_CLUSTER)
        elif self.function == self.RFLBAT:
            return self.function(gradients, malicious_user, NUM_COMPONENTS)
        else:
            raise ValueError("The robust mechanism is not implemented yet.")