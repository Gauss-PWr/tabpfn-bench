def evaluate_model(model, X_test, y_test, metric, device=None, use_tensor=False):
    
    if use_tensor:
        X_test, y_test = (
            torch.tensor(X_test),
            torch.tensor(y_test),
        )
        X_test, y_test = X_test.float(), y_test.long()
        
        if 'Classifier' in model.__class__.__name__:
            y_test = y_test.int()

        if device is not None:
            X_test = X_test.to(device)
            y_test = y_test.to(device)  
    
    pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        pred = model.predict_proba(X_test)
        if pred.shape[1] == 2:
            pred = pred[:, 1]
    else:
        pred = model.predict(X_test)
        
    return metric(y_test, pred)

