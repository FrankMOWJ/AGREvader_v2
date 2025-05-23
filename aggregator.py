import numpy as np
from constants import *
from sklearn.cluster import KMeans
import torch.nn.functional as F
import hdbscan
import time

class Aggregator:
    """
    The aggregator class collecting gradients calculated by participants and plus together
    """

    def __init__(self, sample_gradients: torch.Tensor, device, robust_mechanism=None):
        """
        Initiate the aggregator according to the tensor size of a given sample
        :param sample_gradients: The tensor size of a sample gradient
        :param robust_mechanism: the robust mechanism used to aggregate the gradients
        """
        self.DEVICE = device
        self.sample_gradients = sample_gradients.to(self.DEVICE)
        self.collected_gradients = []
        self.counter = 0
        self.counter_by_indices = torch.ones(self.sample_gradients.size()).to(self.DEVICE)
        self.robust = RobustMechanism(robust_mechanism)

        # AGR related parameters
        self.agr_model = None #Global model Fang required

    def reset(self):
        """
        Reset the aggregator to 0 before next round of aggregation
        """
        self.collected_gradients = []
        self.counter = 0
        self.counter_by_indices = torch.ones(self.sample_gradients.size()).to(self.DEVICE)
        self.agr_model_calculated = False

    def collect(self, gradient: torch.Tensor,indices=None, sample_count=None, source=None):
        """
        Collect one set of gradients from a participant
        :param gradient: The gradient calculated by a participant
        :param souce: The source of the gradient, used for AGR verification
        :param indices: the indices of the gradient, used for AGR verification

        """
        if sample_count is None:
            self.collected_gradients.append(gradient)
            if indices is not None:
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
            # print(f"*******************{NUMBER_OF_ADVERSARY}*******************")
            # if self.robust.type == MULTI_KRUM:
            #     result = self.robust.getter(self.collected_gradients, malicious_user=0)
            # else:

            # 记录时间
            start_time = time.time()
            result = self.robust.getter(self.collected_gradients, malicious_user=NUMBER_OF_ADVERSARY)
            end_time = time.time()
            print(f'defense time:{end_time - start_time}********************')
            # 写入文件
            with open(f"defense_time.txt", "a") as f:
                f.write(f'{end_time - start_time}\n')
      
        
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
        elif robust_mechanism == ANGLE_MEDIAN:
            self.function = self.angle_median
        elif robust_mechanism == ANGLE_TRIM:
            self.function = self.angle_trim
        elif robust_mechanism == TOPK:
            self.function = self.topk
        elif robust_mechanism == DIFFERENTIAL_PRIVACY:
            self.function = self.differential_privacy

        self.agr_model = None
        self.history_gradients = []  # To store historical gradients for each client


    def agr_model_acquire(self, model: torch.nn.Module):
        """
        Acquire the model used for LRR and ERR verification in Fang Defense
        The model must have the same parameters as the global model
        :param model: The model used for LRR and ERR verification
        """
        self.agr_model = model

    def differential_privacy(self, input_gradients: torch.Tensor):
        """
        The differential privacy mechanism
        :param input_gradients: The gradients collected from participants
        :return: The average of the gradients with differential privacy
        """
        # add gaussian noise
        noise = torch.normal(mean=0, std=0.001, size=input_gradients[0].size()).to(input_gradients[0].device)
        
        # for g in input_gradients:
        #     print(g)
        #     print(g + noise)
        #     print("******")
        # print("***************")
            
        return torch.mean(input_gradients + noise, 0)

    def topk(self, input_gradients: torch.Tensor):
        """
        The top-k mechanism
        :param input_gradients: The gradients collected from participants
        :return: The average of the gradients after removing the top-k dimensions
        """
        # for each input gradient, select the top-k dimensions with the largest absolute values
        k = 2
        top_k = len(input_gradients[0]) // k
        top_k_indices = torch.topk(torch.abs(input_gradients), top_k, dim=1).indices
        
        # 创建一个与 input_gradients 形状相同的全零张量
        filtered_gradients = torch.zeros_like(input_gradients).to(input_gradients[0].device)

        # 在新张量中保留 top-k 维度的值，其余维度保持为 0
        for i in range(input_gradients.size(0)):
            filtered_gradients[i][top_k_indices[i]] = input_gradients[i][top_k_indices[i]]

        return torch.mean(filtered_gradients, 0)
                              

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

    def angle_median(self, input_gradients: torch.Tensor):
        num_gradients = input_gradients.size(0) 
        angles = torch.zeros(num_gradients, num_gradients)

        # 计算每一对梯度之间的角度
        for i in range(num_gradients):
            for j in range(i + 1, num_gradients):
                g1 = input_gradients[i]
                g2 = input_gradients[j]

                cos_theta = F.cosine_similarity(g1, g2, dim=0).item()
                # 计算角度
                cos_theta_clamped = max(-1.0, min(1.0, cos_theta))
                theta = torch.acos(torch.tensor(cos_theta_clamped))  # 转为张量计算
                angles[i, j] = theta
                angles[j, i] = theta  # 对称矩阵

        # 计算每个梯度的平均角度
        mean_angles = angles.mean(dim=1)

        # 找到平均角度最小的梯度索引
        min_angle_idx = torch.argmin(mean_angles).item()

        # 返回该梯度
        return input_gradients[min_angle_idx]
    
    def angle_trim(self, input_gradients: torch.Tensor, trim_bound=ANGLE_TRIM_BOUND):
        
        num_gradients = input_gradients.size(0) 
        angles = torch.zeros(num_gradients, num_gradients)

        # 计算每一对梯度之间的角度
        for i in range(num_gradients):
            for j in range(i + 1, num_gradients):
                g1 = input_gradients[i]
                g2 = input_gradients[j]

                cos_theta = F.cosine_similarity(g1, g2, dim=0).item()
                # 计算角度
                cos_theta_clamped = max(-1.0, min(1.0, cos_theta))
                theta = torch.acos(torch.tensor(cos_theta_clamped))
                angles[i, j] = theta
                angles[j, i] = theta  # 对称矩阵
                
        # 计算每个梯度的平均角度
        mean_angles = angles.mean(dim=1)
        
        # 去除最后2*trim_bound个角度最大的梯度
        sorted_indices = torch.argsort(mean_angles)
        assert num_gradients >= 2 * trim_bound
        selected_indices = sorted_indices[:- 2 * trim_bound]
        
        
        # 返回这些梯度的平均值
        return input_gradients[selected_indices].mean(0)
    
    
    def trimmed_mean(self, input_gradients: torch.Tensor, malicious_user: int, trim_bound=TRIM_BOUND):
        """
        The trimmed mean AGR
        :param input_gradients: The gradients collected from participants
        :param malicious_user: The number of malicious participants
        :param trim_bound: The number of gradients to be trimmed from both ends
        :return: The trimmed mean of the gradients
        """
        input_gradients = torch.sort(input_gradients, dim=0).values
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
    
    def multi_krum(self, input_gradients: torch.Tensor, malicious_user: int, num_select=MULTI_KRUM_K):
        """
        The multi-krum method 
        :param input_gradients: The gradients collected from participants
        :param malicious_user: The number of malicious participants
        :return: The average of the gradients after removing the malicious participants
        """
        # multi_k =  (self.type == MULTI_KRUM)
        candidates = []
        candidate_indices = []
        remaining_updates = input_gradients
        all_indices = np.arange(len(input_gradients))

        while len(remaining_updates) > 2 * malicious_user + 2:
            distances = []
            for update in remaining_updates:
                distance = []
                for update_ in remaining_updates:
                    distance.append(torch.norm((update - update_)) ** 2)
                distance = torch.Tensor(distance).float()
                distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

            distances = torch.sort(distances, dim=1)[0]
            scores = torch.sum(distances[:, :len(remaining_updates) - 2 - malicious_user], dim=1)
            indices = torch.argsort(scores)[:len(remaining_updates) - 2 - malicious_user]

            candidate_indices.append(all_indices[indices[0]])
            all_indices = np.delete(all_indices, indices[0])
            candidates = remaining_updates[indices[0]][None, :] if not len(candidates) else torch.cat(
                (candidates, remaining_updates[indices[0]][None, :]), 0)
            remaining_updates = torch.cat((remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)
            # if not multi_k:
            #     break
            # Check if the number of selected gradients reaches the required limit
            if len(candidates) >= num_select:
                break


        print("Selected candicates = {}".format(np.array(candidate_indices)))
        # RobustMechanism.appearence_list = candidate_indices
        return torch.mean(candidates, dim=0)
    
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
    
    def cos_sim_nd(self, p: torch.Tensor, q: torch.Tensor) -> float:
        '''
        Calculate the cosine similarity between two gradient vectors.
        :param p: The first gradient vector.
        :param q: The second gradient vector.
        :return: The cosine similarity.
        '''
        return 1 - (torch.dot(p, q) / (p.norm() * q.norm())).item()
    
    def FLAME(self, input_gradients: torch.Tensor, malicious_user: int, noise_std=0.001) -> torch.Tensor:
        '''
        The FLAME mechanism
        :param input_gradients: The gradients collected from participants
        :param malicious_user: The number of malicious participants
        :return: The average of the gradients after removing the malicious participants
        '''
        num_worker = input_gradients.shape[0]

        # 计算所有客户端更新的余弦相似度矩阵
        cos_sims = np.zeros((num_worker, num_worker))
        for i in range(num_worker):
            for j in range(num_worker):
                cos_sims[i, j] = self.cos_sim_nd(input_gradients[i], input_gradients[j])

        # 使用 HDBSCAN 聚类来过滤恶意客户端
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=int(num_worker / 2) + 1,
            min_samples=1,
            metric='euclidean',  # 使用欧氏距离作为度量标准
            allow_single_cluster=True
        )
        clusterer.fit(cos_sims)  # 聚类
        labels = clusterer.labels_  # 获取每个客户端的标签

        # 找到最大聚类并过滤可信客户端
        max_label = max(set(labels), key=list(labels).count)
        filtered_indices = [i for i, label in enumerate(labels) if label == max_label]

        # 计算每个梯度的L2范数，并计算中位数
        norms = torch.stack([input_gradients[i].norm() for i in filtered_indices])
        median_norm = norms.median()

        # 对可信客户端的梯度进行裁剪
        clipped_gradients = [
            input_gradients[i] * min(1, median_norm / input_gradients[i].norm())
            for i in filtered_indices
        ]
        aggregated_gradient = torch.mean(torch.stack(clipped_gradients), dim=0)

        # 添加高斯噪声
        noise = torch.normal(0, noise_std * median_norm, size=aggregated_gradient.shape).to(aggregated_gradient.device)
        aggregated_gradient += noise

        return aggregated_gradient
    
    #! 有问题
    def Foolsgold(self, input_gradients: torch.Tensor, malicious_user: int):
        '''
        The Foolsgold mechanism
        :param input_gradients: The gradients collected from participants, shape (num_participants, num_params)
        :param malicious_user: The number of malicious participants
        :return: The average of the gradients after removing the malicious participants
        '''
        
        # 1. Update historical gradients
        self.history_gradients.append(input_gradients)
        # Limit the history length
        if len(self.history_gradients) > 50:
            self.history_gradients.pop(0)  # Remove the oldest gradient if history is full

        num_participants = input_gradients.shape[0]
        
        history_tensor = torch.stack(self.history_gradients).to(DEVICE)

        # 2. Compute cosine similarity between pair-wise historical updates
        history_cos_sim = F.cosine_similarity(history_tensor.unsqueeze(1), history_tensor.unsqueeze(2), dim=3).to(self.DEVICE)

        # 3. Initialize credit scores
        credit_scores = torch.ones(num_participants, device=self.DEVICE)

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
        if self.function == self.naive_average or self.function == self.median or self.function == self.angle_median \
            or self.function == self.differential_privacy or self.function == self.topk:
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
        elif self.function == self.angle_trim:
            return self.function(gradients, ANGLE_TRIM_BOUND)
        else:
            raise ValueError("The robust mechanism is not implemented yet.")