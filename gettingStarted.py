import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import ElasticNet
# from sklearn.linear_model import Lasso
# from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# =============================================================================
def main():
    # Read the original data files
    trainDF = pd.read_csv("data/train.csv")
    testDF = pd.read_csv("data/test.csv")
    
    # IMPORTANT - GIVES US INFORMATION ON THE DATA
    # demonstrateHelpers(trainDF)
    
    trainInput, testInput, trainOutput, testIDs, predictors = transformData(trainDF, testDF)
    
    #Not sure if this is working correctly - finds the most accurate k value
    # paramSearchPlot(trainInput, trainOutput)
    
    doExperiment(trainInput, trainOutput, predictors)
    
    # doKaggleTest(trainInput, testInput, trainOutput, testIDs, predictors)

    
# ===============================================================================
'''
Does k-fold CV on the Kaggle training set using LinearRegression.
(You might review the discussion in hw06 about the so-called "Kaggle training set"
versus other sets.)
'''
def doExperiment(trainInput, trainOutput, predictors):
    alg = LinearRegression()
    # alg = KNeighborsClassifier(n_neighbors = 2)
    # alg = ElasticNet()
    # alg = Lasso()
    # alg = BayesianRidge()
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()
    print("CV Average Score:", cvMeanScore)
    
    
# ===============================================================================
'''
Runs the algorithm on the testing set and writes the results to a csv file.
'''
def doKaggleTest(trainInput, testInput, trainOutput, testIDs, predictors):
    alg = LinearRegression()

    # Train the algorithm using all the training data
    alg.fit(trainInput.loc[:, predictors], trainOutput)

    # Make predictions on the test set.
    predictions = alg.predict(testInput.loc[:, predictors])

    # Create a new dataframe with only the columns Kaggle wants from the dataset.
    submission = pd.DataFrame({
        "Id": testIDs,
        "SalePrice": predictions
    })

    # Prepare CSV
    submission.to_csv('data/testResults.csv', index=False)
    # Now, this .csv file can be uploaded to Kaggle

# ============================================================================
# Data cleaning - conversion, normalization

'''
Pre-processing code will go in this function (and helper functions you call from here).
'''
def transformData(trainDF, testDF):
    # predictors = ['1stFlrSF', '2ndFlrSF']
    # numPredNoPrep = ['MSSubClass', 'LotArea', 'OverallQual'
    #         ,'OverallCond', 'YearBuilt', 'YearRemodAdd'
    #         , '1stFlrSF', '2ndFlrSF'
    #         , 'GrLivArea' 
    #         , 'FullBath'
    #         ,'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd'
    #         ,'Fireplaces' 
    #         , 'WoodDeckSF','OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea'
    #         ,'MiscVal', 'MoSold', 'YrSold']
    # predictors = numPredNoPrep
    
    # numPredNoNull = ['MSSubClass', 'LotArea', 'OverallQual'
    #         ,'OverallCond', 'YearBuilt', 'YearRemodAdd'
    #         ,'BsmtFinSF2'
    #         , 'BsmtUnfSF' 
    #         ,'TotalBsmtSF'
    #         , '1stFlrSF', '2ndFlrSF'
    #         ,'LowQualFinSF'
    #         , 'GrLivArea' 
    #         ,'BsmtFullBath' 
    #         ,'BsmtHalfBath'
    #         , 'FullBath'
    #         ,'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd'
    #         ,'Fireplaces' 
    #         ,'GarageCars'
    #         , 'GarageArea'
    #         , 'WoodDeckSF','OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea'
    #         ,'MiscVal', 'MoSold', 'YrSold']
    
    # predictors = numPredNoNull
    
    predictors = ['MSSubClass', 'LotArea', 'OverallQual','OverallCond', 'YearBuilt', 'YearRemodAdd'
            ,'BsmtFinSF2', 'BsmtUnfSF' ,'TotalBsmtSF', '1stFlrSF', '2ndFlrSF','LowQualFinSF', 'GrLivArea' 
            ,'BsmtFullBath' ,'BsmtHalfBath', 'FullBath','HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd'
            ,'Fireplaces' ,'GarageCars', 'GarageArea', 'WoodDeckSF','OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 
            'ScreenPorch', 'PoolArea','MiscVal', 'MoSold', 'YrSold', 'CentralAir', 'LotShape','Neighborhood', 'BldgType', 
            'LandContour', 'Condition1', 'Condition2', 'Heating', 'HeatingQC', 'Functional']
    
    
    '''
    You'll want to use far more predictors than just these two columns, of course. But when you add
    more, you'll need to do things like handle missing values and convert non-numeric to numeric.
    Other preprocessing steps would likely be wise too, like standardization, get_dummies, 
    or converting or creating attributes based on your intuition about what's relevant in housing prices.
    '''
    
    
    #----------------------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------------------
    #PREPROCESSING NUMERICAL ATTRIBUTES
    #----------------------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------------------
  
    
    
    #test the theory that the problem is attribute that have a null attribute that isn't being picked up
    #set the null attribute to 0 for all of the problem attributes
    #----------------------------------------------------------------------------------------------------
    testDF.loc[:, 'BsmtFinSF2'] = testDF.loc[:, 'BsmtFinSF2'].fillna(0)
    
    testDF.loc[:, 'BsmtUnfSF'] = testDF.loc[:, 'BsmtUnfSF'].fillna(0)
    
    testDF.loc[:, 'TotalBsmtSF'] = testDF.loc[:, 'TotalBsmtSF'].fillna(0)
    
    #Don't think this attribute made a difference at all
    testDF.loc[:, 'LowQualFinSF'] = testDF.loc[:, 'LowQualFinSF'].fillna(0)
    
    #Don't think this attribute made a difference at all - reduced accuracy
    #before:
    #CV Average Score: 0.7908331891074659
    #after:
    #CV Average Score: 0.7907684020191119
    testDF.loc[:, 'BsmtFullBath'] = testDF.loc[:, 'BsmtFullBath'].fillna(0)
    
    #Don't think this attribute made a difference at all - reduced accuracy
    #before:
    #CV Average Score: 0.7907684020191119
    #after:
    #CV Average Score: 0.7901962022555631
    testDF.loc[:, 'BsmtHalfBath'] = testDF.loc[:, 'BsmtHalfBath'].fillna(0)
    
    #CV Average Score: 0.7958261608786743
    testDF.loc[:, 'GarageCars'] = testDF.loc[:, 'GarageCars'].fillna(0)
    
    #Don't think this attribute made a difference at all - reduced accuracy
    #before:
    #CV Average Score: 0.7958261608786743
    #after:
    #CV Average Score: 0.7938233530155341
    testDF.loc[:, 'GarageArea'] = testDF.loc[:, 'GarageArea'].fillna(0)
    
      
    #STANDARDIZE THE VARIABLES THAT CAN BE STANDARDIZED - start with anything that uses SF 
    # made close to 0 difference
    # colNames = ['BsmtFinSF2', 'BsmtUnfSF' ,'TotalBsmtSF', '1stFlrSF', '2ndFlrSF','LowQualFinSF', 'WoodDeckSF','OpenPorchSF']
    
    #STANDARDIZE
    # trainDF.loc[:, colNames] = trainDF.loc[:, colNames].apply(lambda col: (col-col.mean())/col.std(), axis=0)
    # testDF.loc[:, colNames] = testDF.loc[:, colNames].apply(lambda col: (col-col.mean())/col.std(), axis=0)
    
    # NORMALIZE
    # trainDF.loc[:, colNames] = trainDF.loc[:, colNames].apply(lambda col: (col-col.min())/(col.max()-col.min()), axis=0)
    # testDF.loc[:, colNames] = testDF.loc[:, colNames].apply(lambda col: (col-col.min())/(col.max()-col.min()), axis=0)
    
    #----------------------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------------------
    #PREPROCESSING NON-NUMERICAL ATTRIBUTES
    #----------------------------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------------------------
    #CENTRAl AIR VALUES: NaN, Y, N
    #PREPROCESSING METHOD: NaN = Mode(train), Y=1, N=0
    #WHEN CHANGING TO NUMERIC VALUE - NEED TO CHANGE TEST AND TRAIN DFS
    trainDF.loc[:, 'CentralAir'] = trainDF.loc[:, 'CentralAir'].map(lambda v: 0 if v=="N" else v)
    trainDF.loc[:, 'CentralAir'] = trainDF.loc[:, 'CentralAir'].map(lambda v: 1 if v=="Y" else v)
    trainDF.loc[:, 'CentralAir'] = trainDF.loc[:, 'CentralAir'].fillna(trainDF.loc[:, 'CentralAir'].mode().iloc[0])
    
    testDF.loc[:, 'CentralAir'] = testDF.loc[:, 'CentralAir'].map(lambda v: 0 if v=="N" else v)
    testDF.loc[:, 'CentralAir'] = testDF.loc[:, 'CentralAir'].map(lambda v: 1 if v=="Y" else v)
    testDF.loc[:, 'CentralAir'] = testDF.loc[:, 'CentralAir'].fillna(trainDF.loc[:, 'CentralAir'].mode().iloc[0])
    # targetDF.loc[:, 'Embarked'] = targetDF.loc[:, 'Embarked'].fillna(sourceDF.loc[:, 'Embarked'].mode().iloc[0])
    
    #----------------------------------------------------------------------------------------------------
    
    #LOT SHAPE VALUES: NaN, 'Reg', 'IR1', 'IR2', 'IR3'
    #PREPROCESSING METHOD: NaN = Mode(train), Reg = 0, IR1 = 1, IR2 = 2, IR3 = 3
    trainDF.loc[:, 'LotShape'] = trainDF.loc[:, 'LotShape'].map(lambda v: 0 if v=="Reg" else v)
    trainDF.loc[:, 'LotShape'] = trainDF.loc[:, 'LotShape'].map(lambda v: 1 if v=="IR1" else v)
    trainDF.loc[:, 'LotShape'] = trainDF.loc[:, 'LotShape'].map(lambda v: 2 if v=="IR2" else v)
    trainDF.loc[:, 'LotShape'] = trainDF.loc[:, 'LotShape'].map(lambda v: 3 if v=="IR3" else v)
    trainDF.loc[:, 'LotShape'] = trainDF.loc[:, 'LotShape'].fillna(trainDF.loc[:, 'LotShape'].mode().iloc[0])
    
    testDF.loc[:, 'LotShape'] = testDF.loc[:, 'LotShape'].map(lambda v: 0 if v=="Reg" else v)
    testDF.loc[:, 'LotShape'] = testDF.loc[:, 'LotShape'].map(lambda v: 1 if v=="IR1" else v)
    testDF.loc[:, 'LotShape'] = testDF.loc[:, 'LotShape'].map(lambda v: 2 if v=="IR2" else v)
    testDF.loc[:, 'LotShape'] = testDF.loc[:, 'LotShape'].map(lambda v: 3 if v=="IR3" else v)
    testDF.loc[:, 'LotShape'] = testDF.loc[:, 'LotShape'].fillna(trainDF.loc[:, 'LotShape'].mode().iloc[0])
    
    #----------------------------------------------------------------------------------------------------
    #NEIGHBORHOOD VALUES: NaN, 'CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Somerst', 'NWAmes', 
    #                       'OldTown', 'BrkSide', 'Sawyer', 'NridgHt', 'NAmes', SawyerW', 'IDOTRR', 'MeadowV', 
    #                       'Edwards', 'Timber', 'Gilbert', 'StoneBr', 'ClearCr', 'NPkVill', 'Blmngtn', 'BrDale', 
    #                       'SWISU', 'Blueste'
    #PREPROCESSING METHOD: NaN = Mode(train), [0-24]
    # df = pd.get_dummies(df, columns=['B'])
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 0 if v=='CollgCr' else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 1 if v=='Veenker' else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 2 if v=='Crawfor' else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 3 if v=='NoRidge' else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 4 if v=='Mitchel' else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 5 if v=='Somerst' else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 6 if v=='NWAmes' else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 7 if v=='OldTown' else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 8 if v=='BrkSide' else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 9 if v=='Sawyer' else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 10 if v=="NridgHt" else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 11 if v=="NAmes" else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 12 if v=="SawyerW" else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 13 if v=="IDOTRR" else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 14 if v=="MeadowV" else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 15 if v=="Edwards" else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 16 if v=="Timber" else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 17 if v=="Gilbert" else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 18 if v=="StoneBr" else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 19 if v=="ClearCr" else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 20 if v=="NPkVill" else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 21 if v=="Blmngtn" else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 22 if v=="BrDale" else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 23 if v=="SWISU" else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].map(lambda v: 24 if v=="Blueste" else v)
    trainDF.loc[:, 'Neighborhood'] = trainDF.loc[:, 'Neighborhood'].fillna(trainDF.loc[:, 'Neighborhood'].mode().iloc[0])
    
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 0 if v=='CollgCr' else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 1 if v=='Veenker' else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 2 if v=='Crawfor' else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 3 if v=='NoRidge' else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 4 if v=='Mitchel' else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 5 if v=='Somerst' else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 6 if v=='NWAmes' else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 7 if v=='OldTown' else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 8 if v=='BrkSide' else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 9 if v=='Sawyer' else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 10 if v=="NridgHt" else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 11 if v=="NAmes" else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 12 if v=="SawyerW" else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 13 if v=="IDOTRR" else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 14 if v=="MeadowV" else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 15 if v=="Edwards" else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 16 if v=="Timber" else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 17 if v=="Gilbert" else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 18 if v=="StoneBr" else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 19 if v=="ClearCr" else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 20 if v=="NPkVill" else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 21 if v=="Blmngtn" else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 22 if v=="BrDale" else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 23 if v=="SWISU" else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].map(lambda v: 24 if v=="Blueste" else v)
    testDF.loc[:, 'Neighborhood'] = testDF.loc[:, 'Neighborhood'].fillna(trainDF.loc[:, 'Neighborhood'].mode().iloc[0])
    #----------------------------------------------------------------------------------------------------
    #BLDGTYPE VALUES: NaN,'1Fam', '2fmCon', 'Duplex', 'TwnhsE', 'Twnhs'
    #PREPROCESSING METHOD: NaN = Mode(train), [0-4]
    trainDF.loc[:, 'BldgType'] = trainDF.loc[:, 'BldgType'].map(lambda v: 0 if v=="1Fam" else v)
    trainDF.loc[:, 'BldgType'] = trainDF.loc[:, 'BldgType'].map(lambda v: 1 if v=="2fmCon" else v)
    trainDF.loc[:, 'BldgType'] = trainDF.loc[:, 'BldgType'].map(lambda v: 2 if v=="Duplex" else v)
    trainDF.loc[:, 'BldgType'] = trainDF.loc[:, 'BldgType'].map(lambda v: 3 if v=="TwnhsE" else v)
    trainDF.loc[:, 'BldgType'] = trainDF.loc[:, 'BldgType'].map(lambda v: 4 if v=="Twnhs" else v)
    trainDF.loc[:, 'BldgType'] = trainDF.loc[:, 'BldgType'].fillna(trainDF.loc[:, 'BldgType'].mode().iloc[0])
    
    testDF.loc[:, 'BldgType'] = testDF.loc[:, 'BldgType'].map(lambda v: 0 if v=="1Fam" else v)
    testDF.loc[:, 'BldgType'] = testDF.loc[:, 'BldgType'].map(lambda v: 1 if v=="2fmCon" else v)
    testDF.loc[:, 'BldgType'] = testDF.loc[:, 'BldgType'].map(lambda v: 2 if v=="Duplex" else v)
    testDF.loc[:, 'BldgType'] = testDF.loc[:, 'BldgType'].map(lambda v: 3 if v=="TwnhsE" else v)
    testDF.loc[:, 'BldgType'] = testDF.loc[:, 'BldgType'].map(lambda v: 4 if v=="Twnhs" else v)
    testDF.loc[:, 'BldgType'] = testDF.loc[:, 'BldgType'].fillna(trainDF.loc[:, 'BldgType'].mode().iloc[0])
    #----------------------------------------------------------------------------------------------------
    #LAND CONTOUR VALUES: NaN,'Lvl', 'Bnk', 'Low', 'HLS'
    #PREPROCESSING METHOD: NaN = Mode(train), [0-3]
    trainDF.loc[:, 'LandContour'] = trainDF.loc[:, 'LandContour'].map(lambda v: 0 if v=="Lvl" else v)
    trainDF.loc[:, 'LandContour'] = trainDF.loc[:, 'LandContour'].map(lambda v: 1 if v=="Bnk" else v)
    trainDF.loc[:, 'LandContour'] = trainDF.loc[:, 'LandContour'].map(lambda v: 2 if v=="Low" else v)
    trainDF.loc[:, 'LandContour'] = trainDF.loc[:, 'LandContour'].map(lambda v: 3 if v=="HLS" else v)
    trainDF.loc[:, 'LandContour'] = trainDF.loc[:, 'LandContour'].fillna(trainDF.loc[:, 'LandContour'].mode().iloc[0])
    
    testDF.loc[:, 'LandContour'] = testDF.loc[:, 'LandContour'].map(lambda v: 0 if v=="Lvl" else v)
    testDF.loc[:, 'LandContour'] = testDF.loc[:, 'LandContour'].map(lambda v: 1 if v=="Bnk" else v)
    testDF.loc[:, 'LandContour'] = testDF.loc[:, 'LandContour'].map(lambda v: 2 if v=="Low" else v)
    testDF.loc[:, 'LandContour'] = testDF.loc[:, 'LandContour'].map(lambda v: 3 if v=="HLS" else v)
    testDF.loc[:, 'LandContour'] = testDF.loc[:, 'LandContour'].fillna(trainDF.loc[:, 'LandContour'].mode().iloc[0])
    #----------------------------------------------------------------------------------------------------
    #CONDITION 1 VALUES: NaN, 'Norm', 'Feedr', 'PosN', 'Artery', 'RRAe', 'RRNn', 'RRAn', 'PosA', 'RRNe'
    #PREPROCESSING METHOD: NaN = Mode(train), 'Norm' = 0, 'Feedr' = 1, 'PosN' = 2, 'Artery' = 3, 'RRAe' = 4, 
    #                       'RRNn' = 5, 'RRAn' = 6, 'PosA' = 7, 'RRNe' = 8

    trainDF.loc[:, 'Condition1'] = trainDF.loc[:, 'Condition1'].map(lambda v: 0 if v=="Norm" else v)
    trainDF.loc[:, 'Condition1'] = trainDF.loc[:, 'Condition1'].map(lambda v: 1 if v=="Feedr" else v)
    trainDF.loc[:, 'Condition1'] = trainDF.loc[:, 'Condition1'].map(lambda v: 2 if v=="PosN" else v)
    trainDF.loc[:, 'Condition1'] = trainDF.loc[:, 'Condition1'].map(lambda v: 3 if v=="Artery" else v)
    trainDF.loc[:, 'Condition1'] = trainDF.loc[:, 'Condition1'].map(lambda v: 4 if v=="RRAe" else v)
    trainDF.loc[:, 'Condition1'] = trainDF.loc[:, 'Condition1'].map(lambda v: 5 if v=="RRNn" else v)
    trainDF.loc[:, 'Condition1'] = trainDF.loc[:, 'Condition1'].map(lambda v: 6 if v=="RRAn" else v)
    trainDF.loc[:, 'Condition1'] = trainDF.loc[:, 'Condition1'].map(lambda v: 7 if v=="PosA" else v)
    trainDF.loc[:, 'Condition1'] = trainDF.loc[:, 'Condition1'].map(lambda v: 8 if v=="RRNe" else v)
    trainDF.loc[:, 'Condition1'] = trainDF.loc[:, 'Condition1'].fillna(trainDF.loc[:, 'Condition1'].mode().iloc[0])
    
    testDF.loc[:, 'Condition1'] = testDF.loc[:, 'Condition1'].map(lambda v: 0 if v=="Norm" else v)
    testDF.loc[:, 'Condition1'] = testDF.loc[:, 'Condition1'].map(lambda v: 1 if v=="Feedr" else v)
    testDF.loc[:, 'Condition1'] = testDF.loc[:, 'Condition1'].map(lambda v: 2 if v=="PosN" else v)
    testDF.loc[:, 'Condition1'] = testDF.loc[:, 'Condition1'].map(lambda v: 3 if v=="Artery" else v)
    testDF.loc[:, 'Condition1'] = testDF.loc[:, 'Condition1'].map(lambda v: 4 if v=="RRAe" else v)
    testDF.loc[:, 'Condition1'] = testDF.loc[:, 'Condition1'].map(lambda v: 5 if v=="RRNn" else v)
    testDF.loc[:, 'Condition1'] = testDF.loc[:, 'Condition1'].map(lambda v: 6 if v=="RRAn" else v)
    testDF.loc[:, 'Condition1'] = testDF.loc[:, 'Condition1'].map(lambda v: 7 if v=="PosA" else v)
    testDF.loc[:, 'Condition1'] = testDF.loc[:, 'Condition1'].map(lambda v: 8 if v=="RRNe" else v)
    testDF.loc[:, 'Condition1'] = testDF.loc[:, 'Condition1'].fillna(trainDF.loc[:, 'Condition1'].mode().iloc[0])
    #----------------------------------------------------------------------------------------------------
    #CONDITION 2 VALUES: NaN, 'Norm', 'Feedr', 'PosN', 'Artery', 'RRAe', 'RRNn', 'RRAn', 'PosA', 'RRNe'
    #PREPROCESSING METHOD: NaN = Mode(train), 'Norm' = 0, 'Feedr' = 1, 'PosN' = 2, 'Artery' = 3, 'RRAe' = 4, 
    #                       'RRNn' = 5, 'RRAn' = 6, 'PosA' = 7

    trainDF.loc[:, 'Condition2'] = trainDF.loc[:, 'Condition2'].map(lambda v: 0 if v=="Norm" else v)
    trainDF.loc[:, 'Condition2'] = trainDF.loc[:, 'Condition2'].map(lambda v: 1 if v=="Feedr" else v)
    trainDF.loc[:, 'Condition2'] = trainDF.loc[:, 'Condition2'].map(lambda v: 2 if v=="PosN" else v)
    trainDF.loc[:, 'Condition2'] = trainDF.loc[:, 'Condition2'].map(lambda v: 3 if v=="Artery" else v)
    trainDF.loc[:, 'Condition2'] = trainDF.loc[:, 'Condition2'].map(lambda v: 4 if v=="RRAe" else v)
    trainDF.loc[:, 'Condition2'] = trainDF.loc[:, 'Condition2'].map(lambda v: 5 if v=="RRNn" else v)
    trainDF.loc[:, 'Condition2'] = trainDF.loc[:, 'Condition2'].map(lambda v: 6 if v=="RRAn" else v)
    trainDF.loc[:, 'Condition2'] = trainDF.loc[:, 'Condition2'].map(lambda v: 7 if v=="PosA" else v)
    trainDF.loc[:, 'Condition2'] = trainDF.loc[:, 'Condition2'].fillna(trainDF.loc[:, 'Condition2'].mode().iloc[0])
    
    testDF.loc[:, 'Condition2'] = testDF.loc[:, 'Condition2'].map(lambda v: 0 if v=="Norm" else v)
    testDF.loc[:, 'Condition2'] = testDF.loc[:, 'Condition2'].map(lambda v: 1 if v=="Feedr" else v)
    testDF.loc[:, 'Condition2'] = testDF.loc[:, 'Condition2'].map(lambda v: 2 if v=="PosN" else v)
    testDF.loc[:, 'Condition2'] = testDF.loc[:, 'Condition2'].map(lambda v: 3 if v=="Artery" else v)
    testDF.loc[:, 'Condition2'] = testDF.loc[:, 'Condition2'].map(lambda v: 4 if v=="RRAe" else v)
    testDF.loc[:, 'Condition2'] = testDF.loc[:, 'Condition2'].map(lambda v: 5 if v=="RRNn" else v)
    testDF.loc[:, 'Condition2'] = testDF.loc[:, 'Condition2'].map(lambda v: 6 if v=="RRAn" else v)
    testDF.loc[:, 'Condition2'] = testDF.loc[:, 'Condition2'].map(lambda v: 7 if v=="PosA" else v)
    testDF.loc[:, 'Condition2'] = testDF.loc[:, 'Condition2'].fillna(trainDF.loc[:, 'Condition2'].mode().iloc[0])
    #----------------------------------------------------------------------------------------------------
    #HEATING VALUES: NaN,'GasA', 'GasW', 'Grav', 'Wall', 'OthW', 'Floor'
    #PREPROCESSING METHOD: NaN = Mode(train), 'GasA' = 0, 'GasW' = 1, 'Grav' = 2, 'Wall' = 3, 'OthW' = 4, 'Floor' = 5
    trainDF.loc[:, 'Heating'] = trainDF.loc[:, 'Heating'].map(lambda v: 0 if v=="GasA" else v)
    trainDF.loc[:, 'Heating'] = trainDF.loc[:, 'Heating'].map(lambda v: 1 if v=="GasW" else v)
    trainDF.loc[:, 'Heating'] = trainDF.loc[:, 'Heating'].map(lambda v: 2 if v=="Grav" else v)
    trainDF.loc[:, 'Heating'] = trainDF.loc[:, 'Heating'].map(lambda v: 3 if v=="Wall" else v)
    trainDF.loc[:, 'Heating'] = trainDF.loc[:, 'Heating'].map(lambda v: 4 if v=="OthW" else v)
    trainDF.loc[:, 'Heating'] = trainDF.loc[:, 'Heating'].map(lambda v: 5 if v=="Floor" else v)
    trainDF.loc[:, 'Heating'] = trainDF.loc[:, 'Heating'].fillna(trainDF.loc[:, 'Heating'].mode().iloc[0])
    
    testDF.loc[:, 'Heating'] = testDF.loc[:, 'Heating'].map(lambda v: 0 if v=="GasA" else v)
    testDF.loc[:, 'Heating'] = testDF.loc[:, 'Heating'].map(lambda v: 1 if v=="GasW" else v)
    testDF.loc[:, 'Heating'] = testDF.loc[:, 'Heating'].map(lambda v: 2 if v=="Grav" else v)
    testDF.loc[:, 'Heating'] = testDF.loc[:, 'Heating'].map(lambda v: 3 if v=="Wall" else v)
    testDF.loc[:, 'Heating'] = testDF.loc[:, 'Heating'].map(lambda v: 4 if v=="OthW" else v)
    testDF.loc[:, 'Heating'] = testDF.loc[:, 'Heating'].map(lambda v: 5 if v=="Floor" else v)
    testDF.loc[:, 'Heating'] = testDF.loc[:, 'Heating'].fillna(trainDF.loc[:, 'Heating'].mode().iloc[0])
    
    #----------------------------------------------------------------------------------------------------
    #HEATING QC VALUES: NaN,'Ex', 'Gd', 'TA', 'Fa', 'Po'
    #PREPROCESSING METHOD: NaN = Median(train), 'Ex' = 0, 'Gd' = 1, 'TA' = 2, 'Fa' = 3, 'Po' = 4
    trainDF.loc[:, 'HeatingQC'] = trainDF.loc[:, 'HeatingQC'].map(lambda v: 0 if v=="Ex" else v)
    trainDF.loc[:, 'HeatingQC'] = trainDF.loc[:, 'HeatingQC'].map(lambda v: 1 if v=="Gd" else v)
    trainDF.loc[:, 'HeatingQC'] = trainDF.loc[:, 'HeatingQC'].map(lambda v: 2 if v=="TA" else v)
    trainDF.loc[:, 'HeatingQC'] = trainDF.loc[:, 'HeatingQC'].map(lambda v: 3 if v=="Fa" else v)
    trainDF.loc[:, 'HeatingQC'] = trainDF.loc[:, 'HeatingQC'].map(lambda v: 4 if v=="Po" else v)
    trainDF.loc[:, 'HeatingQC'] = trainDF.loc[:, 'HeatingQC'].fillna(trainDF.loc[:, 'HeatingQC'].median())
    
    testDF.loc[:, 'HeatingQC'] = testDF.loc[:, 'HeatingQC'].map(lambda v: 0 if v=="Ex" else v)
    testDF.loc[:, 'HeatingQC'] = testDF.loc[:, 'HeatingQC'].map(lambda v: 1 if v=="Gd" else v)
    testDF.loc[:, 'HeatingQC'] = testDF.loc[:, 'HeatingQC'].map(lambda v: 2 if v=="TA" else v)
    testDF.loc[:, 'HeatingQC'] = testDF.loc[:, 'HeatingQC'].map(lambda v: 3 if v=="Fa" else v)
    testDF.loc[:, 'HeatingQC'] = testDF.loc[:, 'HeatingQC'].map(lambda v: 4 if v=="Po" else v)
    testDF.loc[:, 'HeatingQC'] = testDF.loc[:, 'HeatingQC'].fillna(trainDF.loc[:, 'HeatingQC'].median())
    #----------------------------------------------------------------------------------------------------
    #FUNCTIONAL VALUES: NaN,'Typ', 'Min1', 'Maj1', 'Min2', 'Mod', 'Maj2', 'Sev'
    #PREPROCESSING METHOD: NaN = Median(train), 'Typ' = 0, 'Min1' = 1, 'Maj1' = 2, 'Min2' = 3, 'Mod' = 4, 'Maj2' = 5, 'Sev' = 6
    trainDF.loc[:, 'Functional'] = trainDF.loc[:, 'Functional'].map(lambda v: 0 if v=="Typ" else v)
    trainDF.loc[:, 'Functional'] = trainDF.loc[:, 'Functional'].map(lambda v: 1 if v=="Min1" else v)
    trainDF.loc[:, 'Functional'] = trainDF.loc[:, 'Functional'].map(lambda v: 2 if v=="Maj1" else v)
    trainDF.loc[:, 'Functional'] = trainDF.loc[:, 'Functional'].map(lambda v: 3 if v=="Min2" else v)
    trainDF.loc[:, 'Functional'] = trainDF.loc[:, 'Functional'].map(lambda v: 4 if v=="Mod" else v)
    trainDF.loc[:, 'Functional'] = trainDF.loc[:, 'Functional'].map(lambda v: 5 if v=="Maj2" else v)
    trainDF.loc[:, 'Functional'] = trainDF.loc[:, 'Functional'].map(lambda v: 6 if v=="Sev" else v)
    trainDF.loc[:, 'Functional'] = trainDF.loc[:, 'Functional'].fillna(trainDF.loc[:, 'Functional'].mode().iloc[0])
    
    testDF.loc[:, 'Functional'] = testDF.loc[:, 'Functional'].map(lambda v: 0 if v=="Typ" else v)
    testDF.loc[:, 'Functional'] = testDF.loc[:, 'Functional'].map(lambda v: 1 if v=="Min1" else v)
    testDF.loc[:, 'Functional'] = testDF.loc[:, 'Functional'].map(lambda v: 2 if v=="Maj1" else v)
    testDF.loc[:, 'Functional'] = testDF.loc[:, 'Functional'].map(lambda v: 3 if v=="Min2" else v)
    testDF.loc[:, 'Functional'] = testDF.loc[:, 'Functional'].map(lambda v: 4 if v=="Mod" else v)
    testDF.loc[:, 'Functional'] = testDF.loc[:, 'Functional'].map(lambda v: 5 if v=="Maj2" else v)
    testDF.loc[:, 'Functional'] = testDF.loc[:, 'Functional'].map(lambda v: 6 if v=="Sev" else v)
    testDF.loc[:, 'Functional'] = testDF.loc[:, 'Functional'].fillna(trainDF.loc[:, 'Functional'].mode().iloc[0])
    
    #
    #----------------------------------------------------------------------------------------------------
    trainInput = trainDF.loc[:, predictors]
    testInput = testDF.loc[:, predictors]
    '''
    Any transformations you do on the trainInput will need to be done on the
    testInput the same way. (For example, using the exact same min and max, if
    you're doing normalization.)
    '''
    
    trainOutput = trainDF.loc[:, 'SalePrice']
    testIDs = testDF.loc[:, 'Id']
    
    return trainInput, testInput, trainOutput, testIDs, predictors
    
# ===============================================================================
'''
Demonstrates some provided helper functions that you might find useful.
'''
def demonstrateHelpers(trainDF):
    print("Attributes with missing values:", getAttrsWithMissingValues(trainDF), sep='\n')
    
    numericAttrs = getNumericAttrs(trainDF)
    print("Numeric attributes:", numericAttrs, sep='\n')
    
    nonnumericAttrs = getNonNumericAttrs(trainDF)
    print("Non-numeric attributes:", nonnumericAttrs, sep='\n')

    print("Values, for each non-numeric attribute:", getAttrToValuesDictionary(trainDF.loc[:, nonnumericAttrs]), sep='\n')
    
    #added code starts here
    print("Attributes containing NaN:", getNaAttrs(trainDF))
    
    problems = ['BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF','LowQualFinSF','BsmtFullBath','BsmtHalfBath','GarageCars', 'GarageArea']
    print("Values, for problem attributes:", getAttrToValuesDictionary(trainDF.loc[:, problems]), sep='\n')

# ===============================================================================
'''
Returns a dictionary mapping an attribute to the array of values for that attribute.
'''
def getAttrToValuesDictionary(df):
    attrToValues = {}
    for attr in df.columns.values:
        attrToValues[attr] = df.loc[:, attr].unique()

    return attrToValues

# ===============================================================================
'''
Returns the attributes with missing values.
'''
def getAttrsWithMissingValues(df):
    valueCountSeries = df.count(axis=0)  # 0 to count down the rows
    numCases = df.shape[0]  # Number of examples - number of rows in the data frame
    missingSeries = (numCases - valueCountSeries)  # A Series showing the number of missing values, for each attribute
    attrsWithMissingValues = missingSeries[missingSeries != 0].index
    return attrsWithMissingValues

# =============================================================================

'''
Returns the numeric attributes.
'''
def getNumericAttrs(df):
    return __getNumericHelper(df, True)

'''
Returns the non-numeric attributes.
'''
def getNonNumericAttrs(df):
    return __getNumericHelper(df, False)

def __getNumericHelper(df, findNumeric):
    isNumeric = df.applymap(np.isreal) # np.isreal is a function that takes a value and returns True (the value is real) or False
                                       # applymap applies the given function to the whole data frame
                                       # So this returns a DataFrame of True/False values indicating for each value in the original DataFrame whether it is real (numeric) or not

    isNumeric = isNumeric.all() # all: For each column, returns whether all elements are True
    attrs = isNumeric.loc[isNumeric==findNumeric].index # selects the values in isNumeric that are <findNumeric> (True or False)
    return attrs

# =============================================================================
'''
Modification of the numeric attributes method
Returns the attributes with NA values.
'''

def getNaAttrs(df):
    return __getNumericHelper(df, False)

def __getNaHelper(df, findNan):
    isNan = df.applymap(np.isnan) # np.isnan is a function that takes a value and returns True (the value is na) or False
                                       # applymap applies the given function to the whole data frame
                                       # So this returns a DataFrame of True/False values indicating for each value in the original DataFrame whether it is na (numeric) or not

    isNan = isNan.all() # all: For each column, returns whether all elements are True
    attrs = isNan.loc[isNan==findNan].index # selects the values in isNumeric that are <findNumeric> (True or False)
    return attrs

'''
Using Ideas from HW 5 to find the best value for K
Kept getting this error: The least populated class in y has only 1 members, which is less than n_splits=10.
'''
def paramSearchPlot(inputDF, outputSeries):
    neighborList = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 40, 50, 60, 80])
    
    accuracies = neighborList.map(lambda row: model_selection.cross_val_score(KNeighborsClassifier(n_neighbors = row), inputDF, outputSeries, cv=10, scoring='r2').mean())
    print(accuracies)

    plt.plot(neighborList, accuracies)
    plt.xlabel('Neighbors')
    plt.ylabel('Accuracy')
    plt.show()
    
    print(neighborList.loc[accuracies.idxmax()])
# =============================================================================
if __name__ == "__main__":
    main()

