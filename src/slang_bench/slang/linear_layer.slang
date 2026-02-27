module linear_layer;

__exported import NeuralNetworks;

public struct LinearLayer<int M, int N> : IModel<float[N], float[M]>
{
    public StructuredBuffer<float> weight;
    public StructuredBuffer<float> bias;
    public RWStructuredBuffer<float> weight_grad;
    public RWStructuredBuffer<float> bias_grad;

    [BackwardDerivative(backward)]
    public float[M] forward(float[N] x)
    {
        float[M] y;
        for (int i = 0; i < M; ++i)
        {
            float sum = bias[i];
            for (int j = 0; j < N; ++j)
                sum += weight[i * N + j] * x[j];
            y[i] = sum;
        }
        return y;
    }

    public void backward(inout DifferentialPair<float[N]> x, float[M] grad)
    {
        var dx = (float[N]).dzero();
        for (int i = 0; i < M; ++i)
        {
            bias_grad[i] += grad[i];
            for (int j = 0; j < N; ++j)
            {
                weight_grad[i * N + j] += x.p[j] * grad[i];
                dx[j] += weight[i * N + j] * grad[i];
            }
        }
        x = diffPair(x.p, dx);
    }
}
