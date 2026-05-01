import sys
from datetime import datetime
from ConfusionMatrix2 import ConfusionMatrix
from EM import EM


def read_lines(filename):
    with open(filename, 'r') as file:
        lines = [line.strip() for line in file if line.strip()]
    return lines


def parse_docs(lines):
    docs = list()
    docs_topics = list()
    for line in lines:
        if line.startswith('<'):
            docs_topics.append(line.replace('<', '').replace('>', '').split('\t')[2:])
        else:
            docs.append(line.split())
    return docs, docs_topics


def main(develop_file, topics_file):
    lines = read_lines(develop_file)
    docs, docs_topics = parse_docs(lines)
    categories = read_lines(topics_file)
    em = EM(docs, categories)
    clusters = em.start()
    confusion_matrix = ConfusionMatrix(clusters, docs_topics, categories)
    confusion_matrix.compute_confusion_matrix()
    confusion_matrix.print_confusion_matrix()
    confusion_matrix.dominant_topic_for_clusters()
    print(confusion_matrix.accuracy())


if __name__ == "__main__":
    start = datetime.now()
    main(sys.argv[1], sys.argv[2])
    print('Total running time: {0}'.format(datetime.now() - start))
