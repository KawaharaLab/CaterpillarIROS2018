from .config import config
import tensorflow as tf
import numpy as np
from collections import abc
import os


np.seterr(all="raise")


class Logger:
    def __init__(self, graph_name: str, item_labels: abc.Iterable, log_dir=None):
        self.__graph_name = graph_name
        self.__item_labels = item_labels

        with self.graph.as_default():
            self.__item = tf.placeholder(tf.float32)
            scalar = tf.summary.scalar(graph_name, self.__item)
            self.__summary = tf.summary.merge([scalar])

        if log_dir is None:
            log_dir = config.log_dir()
        self.__writers = {
            label: tf.summary.FileWriter("{}/{}".format(log_dir, label))
            for label in item_labels
        }

    def log_values(self, label_value_dict: dict, step: int):
        """
            label_value_dict: {label_0: value, label_1: value, ...}
        """
        assert isinstance(label_value_dict, dict)
        assert len(label_value_dict) == len(self.__item_labels)

        for label in self.__item_labels:
            summary = self.session.run(self.__summary, feed_dict={self.__item: label_value_dict[label]})
            self.__writers[label].add_summary(summary, step)

    @property
    def labels(self) -> abc.Iterable:
        return self.__item_labels

    @property
    def session(self) -> tf.Session:
        try:
            return self.__sess
        except AttributeError:
            self.__sess = tf.Session(graph=self.graph)
            return self.__sess

    @property
    def graph(self) -> tf.Graph:
        try:
            return self.__graph
        except AttributeError:
            self.__graph = tf.Graph()
            return self.__graph

    @property
    def summary(self) -> tf.Summary:
        return self.__summary

    @property
    def summary_writers(self) -> dict:
        return self.__writers
