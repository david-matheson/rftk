// #include <boost/test/unit_test.hpp>

// #include "VectorBuffer.h"
// #include "MatrixBuffer.h"
// #include "BufferCollection.h"
// #include "BufferCollectionStack.h"
// #include "SumOfVarianceImpurity.h"
// #include "SplitpointsImpurity.h"


// struct SumOfVarianceFixture {
//     SumOfVarianceFixture()
//     : splitpoint_counts_key("splitpoint_counts")
//     , child_counts_key("child_counts")
//     , left_stats_key("left_stats")
//     , right_stats_key("right_stats")
//     , collection()
//     , stack()
//     {

//         int splitpoint_count_data[] = {3, 1};
//         const VectorBufferTemplate<int> splitpoint_counts = VectorBufferTemplate<int>(&splitpoint_count_data[0], 2);
//         collection.AddBuffer(splitpoint_counts_key, splitpoint_counts);

//         float child_counts_data[] = {3,3, 6,3, 3,3,
//                                     3,3, 6,3, 0,0};
//         const Tensor3BufferTemplate<float> child_counts = Tensor3BufferTemplate<float>(&child_counts_data[0], 2,3,2);
//         collection.AddBuffer(child_counts_key, child_counts);

//         //2 3 4 5 8 10 u=5.3333 s=7.8889
//         //3 4 5 u=4, s=0.667,  2 8 10 u=6.667 s=11.556
//         //3 8 2 u=4.333 s=6.889  4 5 10 u=6.333 s=6.889
//         //2 3 4 u=3 s=0.667  5 8 10 u=7.666 s=4.222

//         float left_stats_data[] = {12,12,50,50, 26,26,154,154, 9,9,29,29,
//                                    12,12,50,50, 26,26,154,154, 12,12,50,50};
//         const Tensor3BufferTemplate<float> left_stats = Tensor3BufferTemplate<float>(&left_stats_data[0], 2,3,4);
//         collection.AddBuffer(left_stats_key, left_stats);

//         float right_stats_data[] = {20,20,168,168, 19,19,141,141, 23,23,189,189,
//                                     20,20,168,168, 19,19,141,141, 23,23,189,189};
//         const Tensor3BufferTemplate<float> right_stats = Tensor3BufferTemplate<float>(&right_stats_data[0], 2,3,4);
//         collection.AddBuffer(right_stats_key, right_stats);

//         stack.Push(&collection);
//     }

//     ~SumOfVarianceFixture()
//     {
//     }

//     const BufferCollectionKey_t splitpoint_counts_key;
//     const BufferCollectionKey_t child_counts_key;
//     const BufferCollectionKey_t left_stats_key;
//     const BufferCollectionKey_t right_stats_key;
//     BufferCollection collection;
//     BufferCollectionStack stack;
// };

// BOOST_FIXTURE_TEST_SUITE( SumOfVarianceTests,  SumOfVarianceFixture )

// BOOST_AUTO_TEST_CASE(test_SumOfVariance_Impurity)
// {
//     const VectorBufferTemplate<int>& splitpoint_counts = stack.GetBuffer< VectorBufferTemplate<int> >(splitpoint_counts_key);
//     const Tensor3BufferTemplate<float>& child_counts = stack.GetBuffer< Tensor3BufferTemplate<float> >(child_counts_key);
//     const Tensor3BufferTemplate<float>& left_stats = stack.GetBuffer< Tensor3BufferTemplate<float> >(left_stats_key);
//     const Tensor3BufferTemplate<float>& right_stats = stack.GetBuffer< Tensor3BufferTemplate<float> >(right_stats_key);

//     SumOfVarianceImpurity<float> ig;

//     BOOST_CHECK_CLOSE(ig.Impurity(0,0, child_counts, left_stats, right_stats), 3.55, 1.0);
//     BOOST_CHECK_CLOSE(ig.Impurity(0,1, child_counts, left_stats, right_stats), 1.7774, 1.0);
//     BOOST_CHECK_CLOSE(ig.Impurity(0,2, child_counts, left_stats, right_stats), 10.888, 1.0);
//     BOOST_CHECK_CLOSE(ig.Impurity(1,0, child_counts, left_stats, right_stats), 3.55, 1.0);
//     BOOST_CHECK_CLOSE(ig.Impurity(1,1, child_counts, left_stats, right_stats), 1.7774, 1.0);
// }

// BOOST_AUTO_TEST_CASE(test_SumOfVariance_SplitpointsImpurity_ProcessStep)
// {
//     SplitpointsImpurity<SumOfVarianceImpurity<float>, int> splitpointsImpurity(splitpoint_counts_key,
//                                                                                 child_counts_key,
//                                                                                 left_stats_key,
//                                                                                 right_stats_key );
//     boost::mt19937 gen(0);
//     splitpointsImpurity.ProcessStep(stack, collection, gen, collection);
//     const MatrixBufferTemplate<float>& impurities =
//             collection.GetBuffer< MatrixBufferTemplate<float> >(splitpointsImpurity.ImpurityBufferId);

//     BOOST_CHECK_CLOSE(impurities.Get(0,0), 3.55, 1.0);
//     BOOST_CHECK_CLOSE(impurities.Get(0,1), 1.7774, 1.0);
//     BOOST_CHECK_CLOSE(impurities.Get(0,2), 10.888, 1.0);
//     BOOST_CHECK_CLOSE(impurities.Get(1,0), 3.55, 1.0);
//     BOOST_CHECK_CLOSE(impurities.Get(1,1), 0.0, 1.0); //beacause splitpoint_count_data[] = {3, 1};
// }

// BOOST_AUTO_TEST_SUITE_END()