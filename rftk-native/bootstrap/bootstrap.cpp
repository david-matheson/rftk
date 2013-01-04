#include <vector>
#include <algorithm>

void setSeed(int seed)
{
    srand(seed);
}

void sampleWithOutReplacement(int *vec, int dim, int samples)
{
    for(int i=0;i<dim;i++)
    {
        vec[i] = 0;
    }

    std::vector<int> sampleIndices(dim);
    for(int i=0; i<dim; i++)
    {
        sampleIndices[i] = i;
    }
    std::random_shuffle(sampleIndices.begin(), sampleIndices.end());

    samples = samples > dim ? dim : samples;
    for(int i=0; i<samples; i++)
    {
        vec[ sampleIndices[i] ] = 1;
    }
}

void sampleWithReplacement(int *vec, int dim, int samples)
{
    for(int i=0;i<dim;i++)
    {
        vec[i] = 0;
    }

    for(int i=0;i<samples;i++)
    {
        int random = 0;
        random =
        (((int) rand() <<  0) & 0x0000FFFF) |
        (((int) rand() << 16) & 0x5FFF0000);

        int sampleIndex = random % dim;
        vec[sampleIndex]++;
    }
}

void sample(int *vec, int dim, int samples, bool withReplacement)
{
    if(withReplacement)
    {
        return sampleWithReplacement(vec, dim, samples);
    }
    else
    {
        return sampleWithOutReplacement(vec, dim, samples);
    }
}


