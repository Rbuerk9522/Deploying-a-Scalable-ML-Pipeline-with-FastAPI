# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
    Model type: Random Forest Classifier

    Framework: Scikit-Learn

    Predict whether an individual earns >50K or <=50K annually based on U.S. Census demographic data

    Created By: Rodney Buerkley with a template from Udacity 

## Intended Use
    This model is designed for educational and experimental use within Udacity’s “Deploying a Scalable ML Pipeline with FastAPI” project.

## Training Data
    This model was trained on a UCI Adult Census Dataset/

## Evaluation Data
    The data was split with 20% saved for testing and the remaining was used for training.

## Metrics
_Please include the metrics used and your model's performance on those metrics._
Precision: 0.7220 | Recall: 0.2712 | F1: 0.3943

## Ethical Considerations
    Features such as sex, race, and native-country were excluded to prevent biases due to social or economic bases.

## Caveats and Recommendations
    For use retraining is recommended on current, representative data.