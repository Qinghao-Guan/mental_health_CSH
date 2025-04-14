import pandas as pd

from relatio.utils import load_roles
from relatio.utils import load_entities
from relatio.narrative_models import NarrativeModel
from relatio.preprocessing import Preprocessor
from nltk.corpus import stopwords
from relatio.utils import prettify
from relatio import build_graph, draw_graph
import pickle as pk


def entity(df, column_name):
    # create a Preprocessor class
    p = Preprocessor(
        spacy_model="en_core_web_sm",
        remove_punctuation=True,
        remove_digits=True,
        lowercase=True,
        lemmatize=True,
        remove_chars=["\"", '-', "^", ".", "?", "!", ";", "(", ")", ",", ":", "\'", "+", "&", "|", "/", "{", "}",
                      "~", "_", "`", "[", "]", ">", "<", "=", "*", "%", "$", "@", "#", "’"],
        stop_words=stopwords.words('english'),
        n_process=-1,
        batch_size=100
    )
    known_entities = p.mine_entities(
        df[column_name],
        clean_entities=True,
        progress_bar=True,
        output_path='output/entities.pkl'
    )

    # print the most common entities (top 10)
    for n in known_entities.most_common(10): print(n)

    known_entities = load_entities('output/entities.pkl')

    top_known_entities = [e[0] for e in list(known_entities.most_common(100)) if e[0] != '']

    return top_known_entities

def narrative(postproc_roles, top_known_entities):
    m = NarrativeModel(
        clustering='kmeans',
        PCA=True,
        UMAP=True,
        roles_considered=['ARG0', 'B-V', 'B-ARGM-NEG', 'ARG1'],
        roles_with_known_entities=['ARG0', 'ARG1'],
        known_entities = top_known_entities,
        assignment_to_known_entities='embeddings',
        roles_with_unknown_entities=['ARG0', 'ARG1'],
        threshold=0.1
    )

    m.fit(postproc_roles, progress_bar=True)

    m.plot_selection_metric(metric='inertia')

    m.plot_clusters(path='output/clusters.pdf')

    m.clusters_to_txt(path='output/clusters.txt')

    narratives = m.predict(postproc_roles, progress_bar=True)

    pretty_narratives = []
    for n in narratives:
        pretty_narratives.append(prettify(n))

    for i in range(10):
        print(postproc_roles[i])
        print(pretty_narratives[i])

    return narratives, m


def visualization(narratives, m):
    G = build_graph(
        narratives,
        top_n=100,
        prune_network=True
    )

    draw_graph(
        G,
        notebook=True,
        show_buttons=True,
        width="2000px",
        height="2000px",
        output_filename='output/network_of_narratives.html'
    )

    # save the narrative model to a pickle file
    with open('output/narrative_model.pkl', 'wb') as f:
        pk.dump(m, f)


if __name__ == '__main__':
    #df = pd.read_csv("D:\PyCharm 2023.1\\new_reddit\datasets\\reddit_posts_processed.csv")
    df = pd.read_csv("D:\PyCharm 2023.1\\new_reddit\datasets\\youtube_processed.csv")
    postproc_roles = load_roles('output/postproc_roles.json')
    top_known_entities = entity(df, "body")
    narratives, m = narrative(postproc_roles, top_known_entities)
    visualization(narratives, m)
