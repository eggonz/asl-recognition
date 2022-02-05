from abc import ABC, abstractmethod


class ImageDataset(ABC):
    @abstractmethod
    def get_data(self):
        """Returns train and test data
        :return: train_images, train_labels, test_images, test_labels
        """

    @abstractmethod
    def get_train_images(self):
        """Train image data"""
        pass

    @abstractmethod
    def get_train_labels(self):
        """Train image labels"""
        pass

    @abstractmethod
    def get_test_images(self):
        """Test image data"""
        pass

    @abstractmethod
    def get_test_labels(self):
        """Test image labels"""
        pass
