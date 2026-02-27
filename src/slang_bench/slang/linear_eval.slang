import linear_layer;

float[M] eval<int M, int N>(LinearLayer<M, N> model, float[N] input)
{
    return model.forward(input);
}
