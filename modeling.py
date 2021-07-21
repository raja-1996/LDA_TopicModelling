%matplotlib inline

import os
import re

import gensim
import gensim.corpora as corpora
import nltk
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess

nltk.download("stopwords")
nltk.download("wordnet")
en_stop = set(nltk.corpus.stopwords.words("english"))
from nltk.stem import WordNetLemmatizer

stemmer = WordNetLemmatizer()
import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm


class LdaModeling:
    def __init__(self, data):

        self.corpus_superlist = data
        # corpus_superlist
        self.corpus = []
        for sublist in self.corpus_superlist:
            for item in sublist:
                self.corpus.append(item)

        self.corpus = data

    def preprocess_text(self, document):
        # Remove all the special characters
        document = re.sub(r"\W", " ", str(document))

        # remove all single characters
        document = re.sub(r"\s+[a-zA-Z]\s+", " ", document)

        # Remove single characters from the start
        document = re.sub(r"\^[a-zA-Z]\s+", " ", document)

        # Substituting multiple spaces with single space
        document = re.sub(r"\s+", " ", document, flags=re.I)

        # Removing prefixed 'b'
        document = re.sub(r"^b\s+", "", document)

        # Converting to Lowercase
        document = document.lower()

        # Lemmatization
        tokens = document.split()
        tokens = [stemmer.lemmatize(word) for word in tokens]
        tokens = [word for word in tokens if word not in en_stop]
        tokens = [word for word in tokens if len(word) > 5]

        return tokens

    def preprocessing(self):

        processed_data = []
        for doc in self.corpus:
            tokens = self.preprocess_text(doc)
            processed_data.append(tokens)

        gensim_dictionary = corpora.Dictionary(processed_data)
        gensim_corpus = [
            gensim_dictionary.doc2bow(token, allow_update=True) for token in processed_data
        ]

        return gensim_corpus, gensim_dictionary

    def modeling(self, gensim_corpus, gensim_dictionary, num_topics):
        lda_model = gensim.models.ldamodel.LdaModel(
            gensim_corpus, num_topics=num_topics, id2word=gensim_dictionary, passes=50
        )
        lda_model.save("gensim_model.gensim")
        return lda_model

    def plotting(self, lda_model, gensim_corpus, gensim_dictionary):
        print("display")
        vis_data = pyLDAvis.gensim.prepare(lda_model, gensim_corpus, gensim_dictionary, sort_topics=False)
        return vis_data

    def performance(self, lda_model, texts, gensim_corpus, gensim_dictionary):
        print("\nPerplexity:", lda_model.log_perplexity(gensim_corpus))

        texts = [self.preprocess_text(doc) for doc in texts]

        coherence_score_lda = CoherenceModel(
            model=lda_model, texts=texts, dictionary=gensim_dictionary, coherence="c_v"
        )
        coherence_score = coherence_score_lda.get_coherence()
        print("\nCoherence Score:", coherence_score)

    def compute_coherence_values(
        self, docs, gensim_corpus, gensim_dictionary, limit, start=2, step=3
    ):
        """
        Compute c_v coherence for various number of topics

        Parameters:
        ----------
        dictionary : Gensim dictionary
        corpus : Gensim corpus
        texts : List of input texts
        limit : Max num of topics

        Returns:
        -------
        model_list : List of LDA topic models
        coherence_values : Coherence values corresponding to the LDA model with respective number of topics
        """
        coherence_values = []
        model_list = []

        texts = [self.preprocess_text(doc) for doc in docs]

        for num_topics in tqdm(range(start, limit, step)):
            model = self.modeling(gensim_corpus, gensim_dictionary, num_topics)
            model_list.append(model)

            coherencemodel = CoherenceModel(
                model=model, texts=texts, dictionary=gensim_dictionary, coherence="c_v"
            )
            coherence_values.append(coherencemodel.get_coherence())

        return model_list, coherence_values

    def plot_coherence_scores_versus_topics(
        self, docs, gensim_corpus, gensim_dictionary, limit, start, step
    ):

        model_list, coherence_values = self.compute_coherence_values(
            docs, gensim_corpus, gensim_dictionary, limit, start, step
        )

        x = range(start, limit, step)
        plt.plot(x, coherence_values)
        plt.xlabel("Num Topics")
        plt.ylabel("Coherence score")
        plt.legend(("coherence_values"), loc="best")
        plt.show()

        for m, cv in zip(x, coherence_values):
            print("Num Topics =", m, " has Coherence Value of", round(cv, 4))

        best_result_index = coherence_values.index(max(coherence_values))
        optimal_model = model_list[best_result_index]

        return model_list, coherence_values, optimal_model

    def pick_best_model(self, model_list, best_result_index):
        optimal_model = model_list[best_result_index]

        return optimal_model

    def convertldaGenToldaMallet(self, mallet_model):
        model_gensim = gensim.models.ldamodel.LdaModel(
            id2word=mallet_model.id2word,
            num_topics=mallet_model.num_topics,
            alpha=mallet_model.alpha,
            eta=0,
        )
        model_gensim.state.sstats[...] = mallet_model.wordtopics
        model_gensim.sync_state()
        return model_gensim

    def show_topics(self, model):
        print(model.show_topics(formatted=False))

    def format_topics_sentences(self, ldamodel, corpus, texts):
        # Init output
        sent_topics_df = pd.DataFrame()

        # Get main topic in each document
        for i, row in enumerate(ldamodel[corpus]):
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = ldamodel.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(
                        pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]),
                        ignore_index=True,
                    )
                else:
                    break
        sent_topics_df.columns = ["Dominant_Topic", "Perc_Contribution", "Topic_Keywords"]

        # Add original text to the end of the output
        contents = pd.Series(texts)
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        return sent_topics_df

    def get_dominant_topic_of_sentence(self, optimal_model, gensim_corpus, data):
        df_topic_sents_keywords = self.format_topics_sentences(
            ldamodel=optimal_model, corpus=gensim_corpus, texts=data
        )

        # Format
        df_dominant_topic = df_topic_sents_keywords.reset_index()
        df_dominant_topic.columns = [
            "Document_No",
            "Dominant_Topic",
            "Topic_Perc_Contrib",
            "Keywords",
            "Text",
        ]

        return df_dominant_topic

    def get_most_representative_documents_for_topics(self, df_topic_sents_keywords, topn=5):

        sent_topics_sorteddf_mallet = pd.DataFrame()

        sent_topics_outdf_grpd = df_topic_sents_keywords.groupby("Dominant_Topic")

        for i, grp in sent_topics_outdf_grpd:
            sent_topics_sorteddf_mallet = pd.concat(
                [
                    sent_topics_sorteddf_mallet,
                    grp.sort_values(["Topic_Perc_Contrib"], ascending=[0]).head(topn),
                ],
                axis=0,
            )

        # Reset Index
        sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

        # Format
        sent_topics_sorteddf_mallet.columns = [
            "Document_No",
            "Dominant_Topic",
            "Topic_Perc_Contrib",
            "Keywords",
            "Text",
        ]

        return sent_topics_sorteddf_mallet

    def get_count_of_documents_for_topic(self, df_topic_sents_keywords):
        # Number of Documents for Each Topic
        topic_counts = df_topic_sents_keywords["Dominant_Topic"].value_counts()

        # Percentage of Documents for Each Topic
        topic_contribution = round(topic_counts / topic_counts.sum(), 4)

        # Topic Number and Keywords
        topic_num_keywords = df_topic_sents_keywords[["Dominant_Topic", "Keywords"]]

        # Concatenate Column wise
        df_dominant_topics = pd.concat(
            [topic_num_keywords, topic_counts, topic_contribution], axis=1
        )

        # Change Column names
        df_dominant_topics.columns = [
            "Dominant_Topic",
            "Topic_Keywords",
            "Num_Documents",
            "Perc_Documents",
        ]

        # Show
        return df_dominant_topics
