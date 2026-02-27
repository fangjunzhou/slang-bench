import linear_layer;

typealias AtomicFloat = Atomic<float>;

float[M] eval<int M, int N>(LinearLayer<M, N> model, float[N] input)
{
    return model.forward(input);
}

void evalBackward<int M, int N>(LinearLayer<M, N> model, float[N] input, float[M] gradOutput)
{
    float[N] gradInput;
    DifferentialPair<float[N]> inputPair = DifferentialPair<float[N]>(input, gradInput);
    model.backward(inputPair, gradOutput);
}

float[M] evalAtomic<int M, int N>(LinearLayerAtomic<M, N> model, float[N] input)
{
    return model.forward(input);
}

void evalBackwardAtomic<int M, int N>(LinearLayerAtomic<M, N> model, float[N] input, float[M] gradOutput)
{
    float[N] gradInput;
    DifferentialPair<float[N]> inputPair = DifferentialPair<float[N]>(input, gradInput);
    model.backward(inputPair, gradOutput);
}
