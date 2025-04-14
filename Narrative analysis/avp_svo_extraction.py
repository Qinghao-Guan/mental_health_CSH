# Catch warnings for an easy ride
from relatio.logging import FileLogger
from relatio.datasets import load_data
logger = FileLogger(level = 'WARNING')
from relatio.preprocessing import Preprocessor
from nltk.corpus import stopwords
from relatio.semantic_role_labeling import SRL


# create a Preprocessor class
p = Preprocessor(
    spacy_model = "en_core_web_sm",
    remove_punctuation = True,
    remove_digits = True,
    lowercase = True,
    lemmatize = True,
    remove_chars = ["\"",'-',"^",".","?","!",";","(",")",",",":","\'","+","&","|","/","{","}",
                    "~","_","`","[","]",">","<","=","*","%","$","@","#","’"],
    stop_words = stopwords.words('english'),
    n_process = -1,
    batch_size = 100
)

# create a SRL class
SRL = SRL(
    path="https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz",
    batch_size=10,
    cuda_device=-1
)


def preprocessor(dataframe, column_name):
    processed_dataframe = p.split_into_sentences(
                                dataframe,
                                column_name = column_name,
                                output_path=None,
                                progress_bar=True
                            )
    return processed_dataframe


def srl(dataframe):

    srl_res = SRL(dataframe['sentence'][0:900],
                  progress_bar=True)

    from relatio import extract_roles

    roles, sentence_index = extract_roles(
        srl_res,
        used_roles=["ARG0", "B-V", "B-ARGM-NEG", "B-ARGM-MOD", "ARG1", "ARG2"],
        only_triplets=True,
        progress_bar=True
    )

    for d in roles[0:20]: print(d)

    postproc_roles = p.process_roles(roles,
                                     max_length=100,
                                     progress_bar=True,
                                     output_path='output/postproc_roles.json')

    return 'Finish extracting semantic roles'


if __name__ == '__main__':
    #df = load_data(dataset = "D:\PyCharm 2023.1\\new_reddit\datasets\\reddit_posts.csv")
    df = load_data(dataset = "D:\PyCharm 2023.1\\new_reddit\datasets\\youtube_processed.csv")
    data = preprocessor(df, "body")
    print(data)
    semantic_roles = srl(data)
