from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score,precision_score,recall_score

def TimeSeriesDT(data, args): # the input should be features and label from the results of CNN
    # now we use whole 58 data to train the CNN
    X = data[0]
    y = data[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=args.splits, random_state=42)
    print(X_train.shape)
    print(X_test.shape)
    shape, _ = X_train.shape #(2,40) (N,F)
    shape1, _ = X_test.shape
    print(len(y_train)) #N

    DT = DecisionTreeClassifier() # gini
    DT.fit(X_train, y_train)
    y_pred = DT.predict(X_test)
    #y_pred = DT.reshape(y_pred, y_test.shape)
    print(len(y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    result = compute_metrics(y_pred, y_test)  # 获得Acc和F1值
    #print(result)
    return result


def acc_and_f1(preds, labels):
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    pre = precision_score(y_true=labels,y_pred=preds,average='macro',zero_division=1)
    recall = recall_score(y_true=labels,y_pred = preds,average='macro')
    print("F1",f1)
    print("pre",pre)
    print("recall",recall)
    return {
        "f1": f1,
        "pre":pre,
        "recall":recall
    }


def compute_metrics(preds, labels):
    return acc_and_f1(preds, labels)