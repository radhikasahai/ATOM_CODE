# MANUSCRIPT TITLE
# Repository description
# Instructions 
1. run enviornment yaml file: 
`conda env create -f environment.yml`
2. activate the enviornmnet: 
`conda activate ATOM_CODE`
3. load in raw datasets (assay data. at minimum need base rdkit smiles col, response col, and possibly compound_id)
4. create datasets: choose features and sampling techniques (if any)
    - if you plan to do any sampling techniques, you must create the full dataset (scaled w standard scaler) as well 
5. run model pipeline 
    - choose model type 
    - choose model params 
    - then you get results 
    - look at results and UQ 

