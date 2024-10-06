from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

def train_svr_model(df):
    X = df[['Open', 'High', 'Low', 'Close']].values
    y = df['Close'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

    svr = SVR(kernel='rbf')
    svr.fit(X_train, y_train)

    predictions = svr.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    return predictions, mse, mae
