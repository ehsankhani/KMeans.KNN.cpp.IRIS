#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
int findDatasetLine(const std::string& filename, const std::vector<double>& dataset) {
    std::ifstream file(filename);
    if (!file) {
        std::cout << "Failed to open file: " << filename << std::endl;
        return -1;
    }

    std::string line;
    int lineIndex = 0;

    while (std::getline(file, line)) {
        // Split the line into individual values
        std::stringstream ss(line);
        std::vector<double> values;
        double value;
        while (ss >> value) {
            values.push_back(value);
            if (ss.peek() == ',') {
                ss.ignore();
            }
        }

        // Compare the values with the target dataset
        if (values == dataset) {
            // Found the line matching the dataset
            return lineIndex;
        }

        lineIndex++;
    }

    // Dataset not found in the file
    return -1;
}
// Abstract distance class
class Distance {
public:
    virtual double calculate(const std::vector<double>& point1, const std::vector<double>& point2) const = 0;
};

// Concrete implementation of Euclidean distance
class EuclideanDistance : public Distance {
public:
    double calculate(const std::vector<double>& point1, const std::vector<double>& point2) const override {
        double sum = 0.0;
        for (size_t i = 0; i < point1.size(); ++i) {
            double diff = point1[i] - point2[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }
};

// Concrete implementation of Cosine distance
class CosineDistance : public Distance {
public:
    double calculate(const std::vector<double>& point1, const std::vector<double>& point2) const override {
        double dotProduct = 0.0;
        double norm1 = 0.0;
        double norm2 = 0.0;

        for (size_t i = 0; i < point1.size(); ++i) {
            dotProduct += point1[i] * point2[i];
            norm1 += point1[i] * point1[i];
            norm2 += point2[i] * point2[i];
        }

        norm1 = std::sqrt(norm1);
        norm2 = std::sqrt(norm2);

        if (norm1 == 0.0 || norm2 == 0.0) {
            return 0.0;
        }

        return dotProduct / (norm1 * norm2);
    }
};

// Abstract clustering class
class Clustering {
protected:
    int k;
    int maxIterations;
    std::vector<std::vector<double>> centroids;
    const Distance& distance;

public:
    Clustering(int k, int maxIterations, const Distance& distance) : k(k), maxIterations(maxIterations), distance(distance) {}

    virtual void fit(const std::vector<std::vector<double>>& data) = 0;

    int predict(const std::vector<double>& point) {
        if (centroids.empty()) {
            std::cout << "Centroids not initialized. Unable to predict." << std::endl;
            return -1;
        }

        const size_t numFeatures = point.size();
        int nearestCentroid = 0;
        double minDistance = distance.calculate(point, centroids[0]);

        for (int centroidIndex = 1; centroidIndex < k; ++centroidIndex) {
            double calculatedDistance = distance.calculate(point, centroids[centroidIndex]);
            if (calculatedDistance < minDistance) {
                minDistance = calculatedDistance;
                nearestCentroid = centroidIndex;
            }
        }

        return nearestCentroid;
    }
};

// Concrete implementation of KMeans clustering
class KMeans : public Clustering {
public:
    KMeans(int k, int maxIterations, const Distance& distance) : Clustering(k, maxIterations, distance) {}

    void fit(const std::vector<std::vector<double>>& data) override {
        if (data.empty()) {
            std::cout << "Empty dataset. Unable to fit KMeans." << std::endl;
            return;
        }

        const size_t numFeatures = data[0].size();
        const size_t numData = data.size();

        centroids.clear();

        // Initialize centroids with random data points
        std::vector<size_t> centroidIndices;
        for (int i = 0; i < k; ++i) {
            size_t index = rand() % numData;
            centroidIndices.push_back(index);
            centroids.push_back(data[index]);
        }

        for (int iteration = 0; iteration < maxIterations; ++iteration) {
            std::vector<std::vector<double>> clusterSum(k, std::vector<double>(numFeatures));
            std::vector<int> clusterCount(k, 0);

            // Assign data points to nearest centroids
            for (size_t dataIndex = 0; dataIndex < numData; ++dataIndex) {
                const std::vector<double>& point = data[dataIndex];
                int nearestCentroid = 0;
                double minDistance = distance.calculate(point, centroids[0]);

                for (int centroidIndex = 1; centroidIndex < k; ++centroidIndex) {
                    double calculatedDistance = distance.calculate(point, centroids[centroidIndex]);
                    if (calculatedDistance < minDistance) {
                        minDistance = calculatedDistance;
                        nearestCentroid = centroidIndex;
                    }
                }

                // Add data point to cluster
                for (size_t featureIndex = 0; featureIndex < numFeatures; ++featureIndex) {
                    clusterSum[nearestCentroid][featureIndex] += point[featureIndex];
                }
                clusterCount[nearestCentroid]++;
            }

            // Update centroids
            for (int centroidIndex = 0; centroidIndex < k; ++centroidIndex) {
                if (clusterCount[centroidIndex] > 0) {
                    for (size_t featureIndex = 0; featureIndex < numFeatures; ++featureIndex) {
                        centroids[centroidIndex][featureIndex] = clusterSum[centroidIndex][featureIndex] / clusterCount[centroidIndex];
                    }
                }
            }
        }
    }
};

// Abstract classification class
class Classification {
protected:
    int k;
    const Distance& distance;
    std::vector<std::vector<double>> data;
    std::vector<int> labels;

public:
    Classification(int k, const Distance& distance) : k(k), distance(distance) {}

    virtual void fit(const std::vector<std::vector<double>>& data, const std::vector<int>& labels) = 0;

    int predict(const std::vector<double>& point) {
        if (data.empty() || labels.empty()) {
            std::cout << "Data or labels not initialized. Unable to predict." << std::endl;
            return -1;
        }

        const size_t numData = data.size();
        std::vector<std::pair<double, int>> distances;

        for (size_t dataIndex = 0; dataIndex < numData; ++dataIndex) {
            const std::vector<double>& dataPoint = data[dataIndex];
            double calculatedDistance = distance.calculate(point, dataPoint);
            distances.push_back(std::make_pair(calculatedDistance, labels[dataIndex]));
        }

        std::sort(distances.begin(), distances.end(), [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
            return a.first < b.first;
        });

        std::vector<int> labelCount(k, 0);

        for (int i = 0; i < k; ++i) {
            int label = distances[i].second;
            labelCount[label]++;
        }

        int maxCount = labelCount[0];
        int predictedLabel = 0;

        for (int i = 1; i < k; ++i) {
            if (labelCount[i] > maxCount) {
                maxCount = labelCount[i];
                predictedLabel = i;
            }
        }

        return predictedLabel;
    }
};

// Concrete implementation of KNN classification
class KNNClassification : public Classification {
public:
    KNNClassification(int k, const Distance& distance) : Classification(k, distance) {}

    void fit(const std::vector<std::vector<double>>& data, const std::vector<int>& labels) override {
        if (data.empty() || data.size() != labels.size()) {
            std::cout << "Invalid input data and labels. Unable to fit KNN." << std::endl;
            return;
        }

        this->data = data;
        this->labels = labels;
    }
};

int main() {
    // Read train data and labels from files
    std::ifstream dataFile("iris.data");
    std::ifstream labelsFile("iris_labels.data");

    std::vector<std::vector<double>> trainData;
    std::vector<int> trainLabels;

    if (dataFile.is_open() && labelsFile.is_open()) {
        std::string line;
        while (std::getline(dataFile, line)) {
            std::istringstream iss(line);
            std::vector<double> dataPoint;
            double value;
            while (iss >> value) {
                dataPoint.push_back(value);
            }
            trainData.push_back(dataPoint);
        }

        while (std::getline(labelsFile, line)) {
            std::istringstream iss(line);
            int label;
            if (iss >> label) {
                trainLabels.push_back(label);
            }
        }

        dataFile.close();
        //labelsFile.close();
    } else {
        std::cout << "Failed to open data files." << std::endl;
        return 1;
    }

    // Create distance objects
    EuclideanDistance euclideanDistance;
    CosineDistance cosineDistance;

    // Create KMeans object using Euclidean distance
    KMeans kmeansEuclidean(2, 100, euclideanDistance);
    kmeansEuclidean.fit(trainData);

    // Create KMeans object using Cosine distance
    KMeans kmeansCosine(2, 100, cosineDistance);
    kmeansCosine.fit(trainData);

    std::vector<double> testData = {4.6,3.1,1.5,0.2};

    // Find dataset line in the file
    int lineNumber = findDatasetLine("iris.data", testData);
    int flowerCode;
    if (lineNumber >= 0 && lineNumber < 50)
        flowerCode = 0;
    else if (lineNumber >= 50 && lineNumber < 100)
        flowerCode = 1;
    else if (lineNumber >= 100 && lineNumber < 150)
        flowerCode = 2;
    else
        std::cout << "Dataset not found in the file." << std::endl;
    // Create KNN classification objects
    KNNClassification knnEuclidean(3, euclideanDistance);
    knnEuclidean.fit(trainData, trainLabels);

    KNNClassification knnCosine(3, cosineDistance);
    knnCosine.fit(trainData, trainLabels);

    // Perform predictions
    int kmeansEuclideanPrediction = kmeansEuclidean.predict(testData);
    int kmeansCosinePrediction = kmeansCosine.predict(testData);
    int knnEuclideanPrediction = knnEuclidean.predict(testData);
    int knnCosinePrediction = knnCosine.predict(testData);

    std::cout << "The Accurate Code is: " << flowerCode << std::endl;
     
    std::cout << "KMeans Euclidean Prediction: " << kmeansEuclideanPrediction << std::endl;
    std::cout << "KMeans Cosine Prediction: " << kmeansCosinePrediction << std::endl;
    std::cout << "KNN Euclidean Prediction: " << knnEuclideanPrediction << std::endl;
    std::cout << "KNN Cosine Prediction: " << knnCosinePrediction << std::endl;

    return 0;
}