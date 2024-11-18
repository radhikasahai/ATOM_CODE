

# MAKE A FUNTION (this creates the splits)
# nekAll = ["2","3","5","9"]
# #nekAll = ["3","5","9"]

# for nek in nekAll:
#     # Get training data
#     data_path = "/p/lustre2/fan4/NEK_data/NEK"+nek+"/scaled_descriptors/"

#     binding_df = pd.read_csv(data_path+"NEK"+nek+"_1_uM_min_50_pct_binding_with_moe_descriptors.csv") 
#     print(binding_df.shape)

#     print(binding_df.active.value_counts())
#     num_gap = (binding_df.loc[binding_df['active']==0].shape[0]) - (binding_df.loc[binding_df['active']==1].shape[0])
#     print(num_gap)
#     num_minority = binding_df.loc[binding_df['active']==1].shape[0]
#     print(num_minority)

#     # Separate majority and minority classes
#     df_majority = binding_df[binding_df['active']==0]
#     df_minority = binding_df[binding_df['active']==1]

#     #=======================
#     # Create 5-fold splits
#     #=======================
#     kf = KFold(n_splits=5, shuffle=True, random_state=0)

#     # majority
#     for i, (_, v_ind) in enumerate(kf.split(df_majority)):
#         df_majority.loc[df_majority.index[v_ind], 'fold'] = f"fold{i+1}"

#     # minority
#     for i, (_, v_ind) in enumerate(kf.split(df_minority)):
#         df_minority.loc[df_minority.index[v_ind], 'fold'] = f"fold{i+1}"


#     print(df_majority['fold'].value_counts())
#     print(df_minority['fold'].value_counts())


#     # Concat
#     all_fold_df = pd.concat([df_majority,df_minority])
#     print(all_fold_df.shape)
#     print(all_fold_df.active.value_counts())


#     # Save to file
#     split_path = "/p/lustre2/fan4/NEK_data/NEK_data_4Berkeley/NEK"+nek
    
#     if not os.path.exists(split_path):
#         os.makedirs(split_path)

#     all_fold_df.to_csv(split_path+"/NEK"+nek+"_1_uM_min_50_pct_binding_5fold_random_imbalanced.csv", index=False)

#Features
if efcp then grab the SMILES features colum then generate ecfp features 
    else use moe features

#sampling 

def scale(df, path):
    keep id columns
    StandardScalcer(on numeric features)
    return scaled_df

def sampling(scaled_df, path):
if SMOTE
if UNDER
    return balenced_df


#break up funtions 
def get_arrays(file_path, df_filename, filename_type=None, save=False, printOut=False):
    """use dataframes to get trainX, trainy, testX, testy out. Optional: save those files to csv
    file_path: directory
    df_filename: dataframe NEK#_binding_moe_{sampling}_df.csv (sampling: scaled, UNDER, SMOTE, ADASYN)
    split dataframe to train and test, and x and y
    save: bool, option to save splits to separate csv files (train X, train y, test X, test y) 
    returns: numpy arrays train X, train y, testX, test y"""
    df = pd.read_csv(file_path+df_filename)
    train_df= df[df['subset']=='train']
    test_df = df[df['subset']=='test']
    train_y = train_df['active'].to_numpy().reshape(-1)
    test_y=test_df['active'].to_numpy().reshape(-1)
    train_x_df = train_df.drop(columns='active')

  
    test_x_df = test_df.drop(columns='active')
    
    train_x_df = train_df.drop(columns='active')
    test_x_df = test_df.drop(columns='active')
    trainX = train_x_df.select_dtypes(include='number').to_numpy()
    testX = test_x_df.select_dtypes(include='number').to_numpy()
    if (printOut): 

        print(f'train X shape: {trainX.shape}, y: {train_y.shape}, test X: {testX.shape}, y:{test_y.shape}')
    if (save and filename_type is not None): 
        trainxdf = pd.DataFrame(trainX)
        trainxdf.to_csv(file_path+filename_type+'_trainX.csv', index=False)
        # train_x_df.to_csv(filename_type+'_trainX.csv', index=False)
        trainy_df = pd.DataFrame(train_y)
        trainy_df.to_csv(file_path+filename_type+'_train_y.csv', index=False) 
        # test_x_df.to_csv(filename_type+'_testX.csv', index=False)
        testxdf = pd.DataFrame(testX)
        testxdf.to_csv(file_path+filename_type+'_testX.csv', index=False)
        testy_df = pd.DataFrame(test_y)
        testy_df.to_csv(file_path+filename_type+'_test_y.csv', index=False) 
        
    return trainX, train_y, testX, test_y
