import matplotlib.pyplot as plt

def plot_anomalies(df, predictions):
    plt.plot(df['datetime'], df['temperature'], label='Température')
    plt.scatter(df['datetime'][predictions==1], df['temperature'][predictions==1], color='red', label='Anomalies')
    plt.legend()
    plt.show()
