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

public struct LinearLayerAtomic<int M, int N> : IModel<float[N], float[M]>
{
    public StructuredBuffer<float> weight;
    public StructuredBuffer<float> bias;
    public RWStructuredBuffer<Atomic<float>[1]> weight_grad;
    public RWStructuredBuffer<Atomic<float>[1]> bias_grad;

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
            bias_grad[i][0].add(grad[i]);
            for (int j = 0; j < N; ++j)
            {
                weight_grad[i * N + j][0].add(x.p[j] * grad[i]);
                dx[j] += weight[i * N + j] * grad[i];
            }
        }
        x = diffPair(x.p, dx);
    }
}

public struct LinearLayerDiff : IDifferentiablePtrType
{
    typealias Differential = This;
    public RWStructuredBuffer<float> weight;
    public RWStructuredBuffer<float> bias;
}

[BackwardDerivative(backward)]
public float[M] forward<int M, int N>(LinearLayerDiff layer, float[N] x)
{
    float[M] y;
    for (int i = 0; i < M; ++i)
    {
        float sum = layer.bias[i];
        for (int j = 0; j < N; ++j)
            sum += layer.weight[i * N + j] * x[j];
        y[i] = sum;
    }
    return y;
}

public void backward<int M, int N>(DifferentialPtrPair<LinearLayerDiff> layer, inout DifferentialPair<float[N]> x, float[M] grad)
{
    var dx = (float[N]).dzero();
    for (int i = 0; i < M; ++i)
    {
        layer.d.bias[i].add(grad[i]);
        for (int j = 0; j < N; ++j)
        {
            layer.d.weight[i * N + j].add(x.p[j] * grad[i]);
            dx[j] += layer.d.weight[i * N + j] * grad[i];
        }
    }
    x = diffPair(x.p, dx);
}
