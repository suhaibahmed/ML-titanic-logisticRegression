import pandas as pd

titanic_train_df = pd.read_csv('train.csv')


def preprocess_data(titanic_df):

        #input  :titanic dataset as a pandas dataframe
        #output :processed pandas dataframe

        #Removing the 'cabin' attribute as it is missing a lot of values
        titanic_df = titanic_df.drop(['Cabin'], axis=1)

        #Filling the non available dates with median ,i.e.,  ~28
        titanic_df["Age"].fillna(titanic_df["Age"].median(skipna=True), inplace=True)

        #Filling the embarked attribute with the mode
        titanic_df["Embarked"].fillna(titanic_df['Embarked'].value_counts().idxmax(), inplace=True)

        #creating categorical values for PClass (3), Embarked (3) and Sex (2)
        titanic_df=pd.get_dummies(titanic_df, columns=["Pclass","Embarked","Sex"])

        return titanic_df

titanic_df = preprocess_data(titanic_df)
