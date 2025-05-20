from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt

class KMeansImpl:
    def __init__(self, max_iterations=100, tolerance=1e-4, random_state=10):
        ## Parameter initialization
        ## max_iterations: Limit the number of iterations of the algorithm
        ## tolerance: Convergence criteria
        ## random_state: Random state initialization seed for random cluster centroid initializations

        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_state = random_state
        np.random.seed(self.random_state)

    def load_image(self, image_name):
        ##Load the image and return the image as a NumPy array
        
        return np.array(Image.open(image_name))

    def centroid_initialization(self, pixels, num_clusters):
        ##Randomly select centroid initial locations based on the random state seed

        num_pixels = pixels.shape[0]
        random_pixel_s = np.random.choice(num_pixels, num_clusters)
        return pixels[random_pixel_s].copy()
    
    def assign_clusters(self, pixels, centroids, normalized_choice):
        ##Assign each pixel in the image to the closest centroid
        ##Manhattan distance is normalized_choice=1 and Euclidean distance is normalized_choice=2
        ##Adjusting for vectorization implementation to speed up algorithm
        
        ##To use vectorization, reshape the pixels/matrices
        expanded_pixels = pixels[:, np.newaxis, :]
        expanded_centroids = centroids[np.newaxis, :, :]

        if(normalized_choice == 1):
            distance = np.sum(np.abs(expanded_pixels - expanded_centroids), axis=2)
        else:
            distance = np.sum((expanded_pixels - expanded_centroids)**2, axis=2)

        cluster_assignment = np.argmin(distance, axis = 1)
        return cluster_assignment
    
    def move_centroids(self, pixels, cluster_assignment, num_clusters):
        ##Using the mean of the clusters, update that cluster centroid
        ##Adjusting for vectorization implementation to speed up algorithm

        ##Initialize the array, divide the cluster assignments into seperate matrices, and count the pixels in each cluster
        centroids = np.zeros((num_clusters, pixels.shape[1]))
        cluster_matrix = np.zeros((pixels.shape[0], num_clusters))
        cluster_matrix[np.arange(pixels.shape[0]), cluster_assignment]=1
        cluster_size = np.sum(cluster_matrix, axis=0)

        ##Find the new centroids
        for j in range(pixels.shape[1]):
            sum_pixels = np.sum(cluster_matrix * pixels[:,j][:,np.newaxis], axis=0)
            centroids[:,j] = np.divide(sum_pixels, cluster_size, out=np.zeros_like(sum_pixels), where=cluster_size!=0)
        
        empty_clusters = np.where(cluster_size == 0)[0].tolist()
        return centroids, empty_clusters

    def empty_clusters(self, pixels, centroids, cluster_assignment, empty_cluster):
        ##Assign empty clusters to the largest exsisting centroid
        ##Adjusting for vectorization implementation to speed up algorithm

        ##Copy in the new centroids to use the latest position
        new_centroids = centroids.copy()

        if empty_cluster:
            ##Find the size of each cluster
            cluster_sizes = np.bincount(cluster_assignment, minlength=len(centroids))
            for empty_pixels in empty_cluster:
                ##Define the largest cluster
                largest_cluster = np.argmax(cluster_sizes)
                ##Define the indicies of the largest cluster and 
                largest_cluster_index = np.where(cluster_assignment == largest_cluster)[0]
                if len(largest_cluster_index) > 0:
                    random_index = np.random.choice(largest_cluster_index)
                    new_centroids[empty_pixels] = pixels[random_index]

        return new_centroids

    def plot_results(self, origional_image, compressed_image, k, normalized_distance, iterations, duration):
        ##Plot the origional and modified image side by side

        ##Determine if the normalized distance is Manhattan or Euclidean
        if(normalized_distance == 1):
            distance = "Manhattan (L1)"
        else:
            distance = "Euclidean (L2)"
        
        plt.figure(figsize=(10,5))

        plt.subplot(1,2,1)
        plt.imshow(origional_image)
        plt.title("Origional Image")
        plt.axis('off')
        
        plt.subplot(1,2,2)
        plt.imshow(compressed_image)
        plt.title("Compressed Image")
        plt.axis('off')

        plt.show()

    def compress(self, pixels, num_clusters, norm_distance=2):
        ##Use K-Means clustering to compress the image

        result = {
            "class": None,
            "centroid": None,
            "img": None,
            "number_of_iterations": None,
            "time_taken": None,
            "additional_args": {}
        }
    
        ##Record the start time of hte algorithm
        start = time.time()

        ##Save the origional image
        origional_shape = pixels.shape

        ##Turn the image into a 2D array
        twoD_pixels = pixels.reshape(-1, 3)

        ##Run centroid_initialization to start
        centroids = self.centroid_initialization(pixels.reshape(-1 ,3), num_clusters)

        ##Iteration #0 and define prev_centroids to determine convergence
        iterations = 0
        prev_centroids = None

        while (iterations < self.max_iterations):
            ##Assign pixels to clusters and update the centroids
            cluster_assignments = self.assign_clusters(twoD_pixels, centroids, norm_distance)
            new_centroids, empty_clusters = self.move_centroids(twoD_pixels, cluster_assignments, num_clusters)

            ##Deal with empty clusters
            if empty_clusters:
                new_centroids = self.empty_clusters(twoD_pixels, new_centroids, cluster_assignments, empty_clusters)

            ##Check for convergence based on the distance the centroids moved
            if prev_centroids is not None:
                ##Manhattan distance
                if norm_distance == 1:
                    delta = np.sum(np.abs(new_centroids - prev_centroids))
                ##Euclidean distance
                else:
                    delta = np.sum((new_centroids - prev_centroids)**2)

                ##Break if the delta does not exceed the tolerance
                if delta < self.tolerance:
                    break

            ##Update for the next round
            prev_centroids = new_centroids.copy()
            centroids = new_centroids.copy()
            iterations = iterations + 1

        ##Reconstruct the compressed image and convert to uint8 for correct image
        compressed_pixels = centroids[cluster_assignments]
        compressed_image = compressed_pixels.reshape(origional_shape)
        compressed_image = np.clip(compressed_image, 0, 255).astype(np.uint8)

        ##Record the end time and find the duration
        end = time.time()
        duration = end - start

        result["class"] = cluster_assignments.reshape(-1,1)
        result["centroid"] = centroids
        result["img"] = compressed_image
        result["number_of_iterations"] = iterations+1
        result["time_taken"] = duration

        return result

    def run(self):
        ##Run and plot everything

        k_values = [5,10,20,30,40]
        normalization = [1,2]
        
        ##Georgia aquarium picture
        image = self.load_image("georgia-aquarium.jpg")

        for norm in normalization:
            if(norm == 1):
                norm_name = "L1"
            else:
                norm_name = "L2"
            print(f"\nRunning {norm_name} normalization")

            for k in k_values:
                print(f"\nProcessing K={k}")
                
                ##Compress the image 
                results = self.compress(image, k, norm)

                ##Plot the results
                self.plot_results(image, results["img"], k, norm, results["number_of_iterations"], results["time_taken"])

                ##Print the number of iterations and execution time of each run
                print(f"Iterations: {results['number_of_iterations']}, Time: {results['time_taken']} seconds")
    


        ##Football picture
        image = self.load_image("football.bmp")

        for norm in normalization:
            if(norm == 1):
                norm_name = "L1"
            else:
                norm_name = "L2"
            print(f"\nRunning {norm_name} normalization")

            for k in k_values:
                print(f"\nProcessing K={k}")
                
                ##Compress the image 
                results = self.compress(image, k, norm)

                ##Plot the results
                self.plot_results(image, results["img"], k, norm, results["number_of_iterations"], results["time_taken"])

                ##Print the number of iterations and execution time of each run
                print(f"Iterations: {results['number_of_iterations']}, Time: {results['time_taken']} seconds")
            
        ##Sunset picture
        image = self.load_image("Sunset.png")

        for norm in normalization:
            if(norm == 1):
                norm_name = "L1"
            else:
                norm_name = "L2"
            print(f"\nRunning {norm_name} normalization")

            for k in k_values:
                print(f"\nProcessing K={k}")
                
                ##Compress the image 
                results = self.compress(image, k, norm)

                ##Plot the results
                self.plot_results(image, results["img"], k, norm, results["number_of_iterations"], results["time_taken"])

                ##Print the number of iterations and execution time of each run
                print(f"Iterations: {results['number_of_iterations']}, Time: {results['time_taken']} seconds")

##Execute the file
kmeans = KMeansImpl(max_iterations=100, tolerance=1e-4, random_state=10)
kmeans.run()
