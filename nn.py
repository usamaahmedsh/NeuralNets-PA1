"""
ISTA 457 / INFO 557 Fall 2024 HW 1
The main code for the Strings-to-Vectors assignment. See README.md for details.
"""
from typing import Sequence, Any

import numpy as np


class Index:
    """
    Represents a mapping from a vocabulary (e.g., strings) to integers.
    """

    def __init__(self, vocab: Sequence[Any], start=0):
        """
        Assigns an index to each unique item in the `vocab` iterable,
        with indexes starting from `start`.

        Indexes should be assigned in order, so that the first unique item in
        `vocab` has the index `start`, the second unique item has the index
        `start + 1`, etc.
        """
        self.index_mapping = {}
        self.start = start

        for item in vocab:
            if item not in self.index_mapping:
                self.index_mapping[item] = start
                start += 1



    def objects_to_indexes(self, object_seq: Sequence[Any]) -> np.ndarray:
        """
        Returns a vector of the indexes associated with the input objects.

        For objects not in the vocabulary, `start-1` is used as the index.

        :param object_seq: A sequence of objects.
        :return: A 1-dimensional array of the object indexes.
        """

        index_array = np.empty(len(object_seq))

        for i, obj in enumerate(object_seq):
            if obj in self.index_mapping:
                index_array[i] = self.index_mapping[obj]
            else:
                index_array[i] = self.start - 1

        return index_array



    def objects_to_index_matrix(
            self, object_seq_seq: Sequence[Sequence[Any]]) -> np.ndarray:
        """
        Returns a matrix of the indexes associated with the input objects.

        For objects not in the vocabulary, `start-1` is used as the index.

        If the sequences are not all of the same length, shorter sequences will
        have padding added at the end, with `start-1` used as the pad value.

        :param object_seq_seq: A sequence of sequences of objects.
        :return: A 2-dimensional array of the object indexes.
        """

        pad_value = self.start - 1

        max_length = max(len(seq) for seq in object_seq_seq)

        matrix_of_index = np.ones((len(object_seq_seq), max_length), dtype = int) * pad_value

        for i, obj_seq in enumerate(object_seq_seq):
            for j, obj in enumerate(obj_seq):
                if obj in self.index_mapping:
                    matrix_of_index[i,j] = self.index_mapping[obj]
                else:
                    matrix_of_index[i,j] = pad_value


        return matrix_of_index



    def objects_to_binary_vector(self, object_seq: Sequence[Any]) -> np.ndarray:
        """
        Returns a binary vector, with a 1 at each index corresponding to one of
        the input objects.

        :param object_seq: A sequence of objects.
        :return: A 1-dimensional array, with 1s at the indexes of each object,
                 and 0s at all other indexes.
        """

        vector_length = max(self.index_mapping.values(), default = 0)

        binary_vector = np.zeros(vector_length + 1)

        for obj in object_seq:
            if obj in self.index_mapping:
                index = self.index_mapping[obj]
                binary_vector[index] = 1

        return binary_vector

    def objects_to_binary_matrix(
            self, object_seq_seq: Sequence[Sequence[Any]]) -> np.ndarray:
        """
        Returns a binary matrix, with a 1 at each index corresponding to one of
        the input objects.

        :param object_seq_seq: A sequence of sequences of objects.
        :return: A 2-dimensional array, where each row in the array corresponds
                 to a row in the input, with 1s at the indexes of each object,
                 and 0s at all other indexes.
        """

        vector_length = max(self.index_mapping.values(), default = 0)

        binary_matrix = np.zeros((len(object_seq_seq), vector_length + 1))

        for i, object_seq in enumerate(object_seq_seq):
            for obj in object_seq:
                if obj in self.index_mapping:
                    index = self.index_mapping[obj]
                    binary_matrix[i, index] = 1

        return binary_matrix

    def indexes_to_objects(self, index_vector: np.ndarray) -> Sequence[Any]:
        """
        Returns a sequence of objects associated with the indexes in the input
        vector.

        If, for any of the indexes, there is not an associated object, that
        index is skipped in the output.

        :param index_vector: A 1-dimensional array of indexes
        :return: A sequence of objects, one for each index.
        """
        reverse_index_mapping = {v: k for k, v in self.index_mapping.items()}
        objects = []

        for index in index_vector:
            if index in reverse_index_mapping:
                objects.append(reverse_index_mapping[index])

        return objects

    def index_matrix_to_objects(
            self, index_matrix: np.ndarray) -> Sequence[Sequence[Any]]:
        """
        Returns a sequence of sequences of objects associated with the indexes
        in the input matrix.

        If, for any of the indexes, there is not an associated object, that
        index is skipped in the output.

        :param index_matrix: A 2-dimensional array of indexes
        :return: A sequence of sequences of objects, one for each index.
        """
        reverse_index_mapping = {v: k for k, v in self.index_mapping.items()}
        object_matrix = []

        for row in index_matrix:
            objects = [reverse_index_mapping.get(index, None) for index in row]
            objects = [obj for obj in objects if obj is not None]
            object_matrix.append(objects)

        return object_matrix

    def binary_vector_to_objects(self, vector: np.ndarray) -> Sequence[Any]:
        """
        Returns a sequence of the objects identified by the nonzero indexes in
        the input vector.

        If, for any of the indexes, there is not an associated object, that
        index is skipped in the output.

        :param vector: A 1-dimensional binary array
        :return: A sequence of objects, one for each nonzero index.
        """
        reverse_index_mapping = {v: k for k, v in self.index_mapping.items()}

        nonzero_indexes = np.nonzero(vector)[0]

        objects = [reverse_index_mapping.get(index, None) for index in nonzero_indexes]

        objects = [obj for obj in objects if obj is not None]

        return objects

    def binary_matrix_to_objects(
            self, binary_matrix: np.ndarray) -> Sequence[Sequence[Any]]:
        """
        Returns a sequence of sequences of objects identified by the nonzero
        indices in the input matrix.

        If, for any of the indexes, there is not an associated object, that
        index is skipped in the output.

        :param binary_matrix: A 2-dimensional binary array
        :return: A sequence of sequences of objects, one for each nonzero index.
        """
        reverse_index_mapping = {v: k for k, v in self.index_mapping.items()}

        object_matrix = []

        for row in binary_matrix:
            nonzero_indexes = np.nonzero(row)[0]
            objects = [reverse_index_mapping.get(index, None) for index in nonzero_indexes]
            objects = [obj for obj in objects if obj is not None]
            object_matrix.append(objects)

        return object_matrix
