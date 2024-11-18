import pyforest

def rf_results(model, x_input, y_labels): 
    """Make predictions adn get probabilities
    @params
    model: fitted model (fitted to train set)
    train_x, train_y, test_x, test_y: train and test set inputs (np arrays)
    @returns
    train/test predictions
    train/test accuracies 
    train/test probabilities"""
    pred = model.predict(x_input)
    tp, tn, fp, fn = calculate_metrics(y_labels, pred)
    acc = accuracy_score(y_labels, pred)
    precision = precision_score(y_labels, pred)
    recall = recall_score(y_labels, pred)
    specificity = specificity_score(tn, fp)
    prob = model.predict_proba(x_input)

    print(f'accuracy: {acc:.3f}, precision: {precision:.3f}, recall: {recall:.3f}, specificity: {specificity:.3f}')
    return {'Prediction': pred,'Accuracy' : acc, 'Precision' :precision, 'Recall' :recall, 'Specificity': specificity,'Probability' :prob}