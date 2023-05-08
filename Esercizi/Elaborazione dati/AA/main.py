from formule import print_tre_volte
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

# xpoints = np.array([0, 6])
# ypoints = np.array([0, 250])

x=[1,2,3,4]
y=[1,4,9,16]

def main():
    # print_tre_volte('bye')
    # plt.plot(xpoints, ypoints)
    # plt.show()
    scatter = go.Scatter(x=x,y=y)
    fig=go.Figure(data=[scatter])
    fig.update_layout(title_text="Grafico1")
    fig.show()
    print("hello word")
    
if __name__=="__main__":
    main()

