from sklearn.ensemble import IsolationForest

def detect_anomalies(df):
    clf = IsolationForest(contamination=0.05)
    clf.fit(df)
    df['anomaly'] = clf.predict(df)
    return df
