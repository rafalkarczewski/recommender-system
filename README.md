# Implementation of baseline model for recommendation system with positive and unknown ratings.
A proposed approach is compared with the approaches described in [Collaborative Recurrent Autoencoder: Recommend while Learning to Fill in the Blanks](https://arxiv.org/abs/1611.00454) and in [Large-Scale Off-Target Identification Using Fast and Accurate Dual Regularized One-Class Collaborative Filtering and Its Application to Drug Repurposing](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005135).
### Setup
1. First, install docker using these [instructions](https://docs.docker.com/install/).
2. Build the docker image by running `make build`. This make take several minutes.
3. To run experiments, type `make notebook`. This starts a jupyter notebook server on `http://localhost:9999`
### Sample usage

    from preprocessing import TextPreprocessor
    from models import BlockCoordinateAscent
    from metrics import recall_lift
    
    prep = TextPreprocessor(glove_components=300, min_df=5, max_df=0.4)
    processed_articles = prep.fit(articles, window=10, epochs=100)
    embedded_articles = prep.idf_embed(processed_articles)
    
    model = BlockCoordinateAscent(train_ones, embedded_articles_idf)
    model.fit(n_epochs=5)
    predictions = model.predict()  # filling the missing values in the recommendation matrix
    
    test_metrics = recall_lift(predictions, train_ones, test_ones, 300)
    print(test_metrics.recall)

#### TODO

1. Unit tests
