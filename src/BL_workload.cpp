#include <bits/stdc++.h>
#include <immintrin.h>
#include <chrono>
#include "BOBHash32.h"
#include "RBF.h"
#include "REncoderBL.h"
using namespace std;

// For pretty print.
static const char *kGreen = "\033[0;32m";
static const char *kWhite = "\033[0;0m";

// Number of queries to RBF.
long long query_count = 0;
// Number of cache hits (the queried prefix is the same as the last queried one).
long long cache_hit = 0;

// Parameters of REncoder.
double BPK;
double SAME_RATE;
int HASH_NUM = 3;
int STORED_LEVELS;
int IS_SELF_ADAPT;
int SELF_ADAPT_STEP;
int START_LEVEL = 1;
int END_LEVEL = -1;
string VERSION;
set<string> version_set = {"REncoder","REncoderSS","REncoderSE"};

YRENCODER rencoder1, rencoder2;

// Parameters of workload.
string KEY_TYPE;
int TOTAL_ITEM_NUM;
int RANGE_QUERY_NUM;

vector<uint64_t> keys;
vector<pair<uint64_t, uint64_t>> range_queries;
vector<vector<pair<uint64_t, uint64_t> > > test_queries;
vector<vector<uint32_t> > res_queries;
vector<pair<uint64_t, uint64_t>> sample_queries;
set<uint64_t> key_set;

// Borrowed from https://github.com/Eric-R-Knorr/Proteus.
std::vector<std::pair<uint64_t, uint64_t>> SampleQueries(double sample_rate)
{
    std::vector<std::pair<uint64_t, uint64_t>> sample_queries;
    std::default_random_engine generator;
    std::bernoulli_distribution distribution(sample_rate);
    for (auto const &q : range_queries)
    {
        if (distribution(generator))
        {
            sample_queries.push_back(q);
        }
    }
    printf("Percent Sample Queries: %lf\n", (sample_queries.size() * 1.0 / range_queries.size()) * 100.0);
    printf("Sample Queries Num: %ld\n", sample_queries.size());
    return sample_queries;
}

// Borrowed from https://github.com/Eric-R-Knorr/Proteus.
int LongestCommonPrefix(const uint64_t a, const uint64_t b, const size_t max_klen)
{
    (void)max_klen;
    uint64_t xored = a ^ b;
    return xored == 0 ? 64 : __builtin_clzll(xored);
}

// Borrowed from https://github.com/Eric-R-Knorr/Proteus.
std::pair<int, int> KQlcp(const std::vector<uint64_t> &keys,
                          const size_t max_klen,
                          std::vector<uint64_t>::const_iterator &kstart,
                          uint64_t qleft,
                          uint64_t qright)
{
    /*
        THIS FUNCTION RETURNS THE BIT LENGTH OF THE LONGEST COMMON PREFIX
        BETWEEN THE QUERY & THE KEY SET.

        Legend:
        QLeft - Left Query Bound
        QRight - Right Query Bound
        KLeft - Largest key that is < QLeft
        KRight - Smallest key that is >= QRight
        KMin - Smallest Key
        KMax - Largest Key
        KStart - Smallest Key that is >= QLeft
    */

    // QLeft <= KStart
    kstart = lower_bound(kstart, keys.cend(), qleft);

    // Negative query should yield: QLeft < QRight <= KStart
    if (kstart != keys.cend() && *kstart <= qright)
    {
        // Positive Query
        // QLeft <= KStart <= QRight
        return std::make_pair(-1, -1);
    }

    // Derive KLeft by taking the key before KStart
    // KLeft < QLeft < QRight <= KStart = KRight
    auto kleft = (kstart != keys.cbegin()) ? kstart - 1 : kstart;
    auto kright = kstart;

    if (qleft > *kleft && kright != keys.cend() && qright < *kright)
    {
        // KLeft QLeft QRight KRight
        return std::make_pair(LongestCommonPrefix(qleft, *kleft, max_klen),
                              LongestCommonPrefix(qright, *kright, max_klen));
    }
    else if (qleft < *kleft && qright < *kleft)
    {
        // QLeft QRight KMin
        return std::make_pair(-1, LongestCommonPrefix(qright, *kleft, max_klen));
    }
    else if (kright == keys.cend() && keys.back() < qleft)
    {
        // KMax QLeft QRight
        return std::make_pair(LongestCommonPrefix(qleft, keys.back(), max_klen), -1);
    }
    else
    {
        assert(false);
    }

    // Should not reach here!
    return std::make_pair(-1, -1);
}

// Select the start and end level of storing according to dataset/workload.
void SetBeginEndLevel()
{
    int max_kklcp = -1;
    int max_kqlcp = -1;
    for (size_t i = 1; i < keys.size(); i++)
    {
        max_kklcp = max(max_kklcp, LongestCommonPrefix(keys[i], keys[i - 1], 64));
    }
    if (VERSION == "REncoderSE")
    {
        sample_queries = SampleQueries(0.001);
        std::vector<uint64_t>::const_iterator kstart = keys.cbegin();
        for (auto const &q : sample_queries)
        {
            std::pair<int, int> lcps = KQlcp(keys, 64, kstart, q.first, q.second);
            if (lcps.first < 0 && lcps.second < 0)
            {
                continue;
            }
            else
            {
                max_kqlcp = max(max_kqlcp, lcps.first);
                max_kqlcp = max(max_kqlcp, lcps.second);
            }
        }
    }
    printf("max_kklcp: %d\n", max_kklcp);
    printf("max_kqlcp: %d\n", max_kqlcp);
    if (VERSION == "REncoderSE" && max_kklcp < max_kqlcp)
    {
        END_LEVEL = 64 - max_kqlcp;
    }
    START_LEVEL = 66 - max_kklcp;
    return;
}

// Convert string to unsigned 64-bit integer.
uint64_t StringToUINT64(const std::string s)
{
    uint64_t out = 0;
    memcpy(&out, s.c_str(), 8);
    out = __builtin_bswap64(out);
    return out;
}

// Load keys from file.
void LoadKey()
{
    ifstream keyFile;
    keyFile.open("../data/key.txt");
    uint64_t key;
    string keystring;
    for (int i = 0; i < TOTAL_ITEM_NUM; i++)
    {
        if (KEY_TYPE == "string")
        {
            keyFile >> keystring;
            key = StringToUINT64(keystring);
        }
        else
        {
            keyFile >> key;
        }
        keys[i] = key;
    }
    sort(keys.begin(), keys.end());
    for (int i = 0; i < TOTAL_ITEM_NUM; i++)
    {
        key_set.insert(keys[i]);
    }
    keyFile.close();
}
// Load range queries from file.
void LoadQuery(double rate, int repeat)
{
    ifstream leftFile, rightFile;
    // Load the left and right boundaries of the range queries separately.
    leftFile.open("../data/lower_bound.txt");
    rightFile.open("../data/upper_bound.txt");
    uint64_t lower;
    string lowerstring;
    uint64_t upper;
    string upperstring;
    int same_num = RANGE_QUERY_NUM * rate;
    for (int i = 0; i < RANGE_QUERY_NUM; i++)
    {
        if (KEY_TYPE == "string")
        {
            leftFile >> lowerstring;
            lower = StringToUINT64(lowerstring);
            rightFile >> upperstring;
            upper = StringToUINT64(upperstring);
        }
        else
        {
            leftFile >> lower;
            rightFile >> upper;
        }
        upper -= 1;
        range_queries[i] = make_pair(lower, upper);
        test_queries[0][i] = range_queries[i];
   
    }
    for (int k = 1; k < repeat; k ++ ) { 
        for (int i = 0; i < RANGE_QUERY_NUM; i++)
        {
            if (rate != 1.0) {
                if (KEY_TYPE == "string")
                {
                    leftFile >> lowerstring;
                    lower = StringToUINT64(lowerstring);
                    rightFile >> upperstring;
                    upper = StringToUINT64(upperstring);
                }
                else
                {
                    leftFile >> lower;
                    rightFile >> upper;
                }

                upper -= 1;
            }
            if(i >= same_num) {
                test_queries[k][i] = make_pair(lower, upper);
            }
            else {
                test_queries[k][i] = test_queries[0][i];
            }
        } 
    }
    // Shuffle to make the queries random.
    for (int k = 0 ; k < repeat; k ++) {
        random_shuffle(test_queries[k].begin(), test_queries[k].end());
    }
    leftFile.close();
    rightFile.close();
} 
// Calculate and print the proportion of empty queries in all queries.
void PrintEmptyRate()
{
    set<uint64_t>::iterator iter;
    int empty_cnt = 0;
    for (int i = 0; i < RANGE_QUERY_NUM; i++)
    {
        iter = key_set.lower_bound(range_queries[i].first);
        if (!(iter != key_set.end() && (*iter) <= range_queries[i].second))
        {
            empty_cnt++;
        }
    }
    printf("Empty Rate: %lf\n", empty_cnt / 1.0 / RANGE_QUERY_NUM);
} 
// Run the workload and print the metrics.
void RunWorkload()
{
    // Number of repetitions.
    int repeat = 2;

    key_set.clear();
    res_queries.resize(repeat);
    test_queries.resize(repeat);
    for (int k = 0 ; k < repeat; k ++)
    {
        test_queries[k].resize(RANGE_QUERY_NUM);
        res_queries[k].resize(RANGE_QUERY_NUM);
    }
    LoadKey();
    LoadQuery(SAME_RATE, repeat);

    PrintEmptyRate();

    // Select the start and end level of storing.
    if (VERSION == "REncoderSS" || VERSION == "REncoderSE")
    {
        SetBeginEndLevel();
    }

    
    // Initialize REncoder.
    uint64_t memory = (uint64_t)TOTAL_ITEM_NUM * BPK;
    
     if (VERSION == "REncoderSS" || VERSION == "REncoderSE") {
        START_LEVEL = 17;
        END_LEVEL = 21;
    }
    rencoder1.init(memory, HASH_NUM, 64, STORED_LEVELS, START_LEVEL, END_LEVEL);
 

    // Insert keys into REncoder.
    if (IS_SELF_ADAPT)
    {
        // Self-adaptively choose the optimal number of stored levels.
        int true_level = rencoder1.Insert_SelfAdapt(keys, SELF_ADAPT_STEP);
    }
    else
    {
        // Choose the number of stored levels given by arguments.
        if (END_LEVEL == -1)
        {
            rencoder1.stored_level = rencoder1.start_level + STORED_LEVELS;
            for (int i = 0; i < TOTAL_ITEM_NUM; i++)
            {
                rencoder1.Insert(keys[i], START_LEVEL);
            }
            rencoder1.stored_level -= 1;
        }
        else
        {
            rencoder1.start_level = max(1, END_LEVEL - STORED_LEVELS + 1);
            rencoder1.stored_level += 1;
            for (int i = 0; i < TOTAL_ITEM_NUM; i++)
            {
                rencoder1.Insert(keys[i], rencoder1.start_level);
            }
            rencoder1.stored_level -= 1;
        }
    } 
    

     for (int k = 0; k < repeat; k++) {
        for (uint32_t i = 0; i < RANGE_QUERY_NUM; i++)
        {
            
            uint64_t l = test_queries[k][i].first, r = test_queries[k][i].second;
            
            bool res = rencoder1.RangeQuery(l, r);
            
            
            auto iter = key_set.lower_bound(l);
            res_queries[k][i] = false;
            if (iter != key_set.end() && (*iter) <= r)
            {
                res_queries[k][i] = true;
                // Test for false negatives.
                if (!res)
                {
                    cout << "Range Query Error (False Negative)";
                    exit(-1);
                } 
                continue;
            }
            
                if (res)
                {
                    
                    rencoder1.add_black_list_node(l,r);
                    
                } 
                        
        } 
     }
    
    uint64_t bl_size = rencoder1.query_black_list_size();
    printf("black list size : %lu , total memory : %lf\n",bl_size, (uint64_t)TOTAL_ITEM_NUM * BPK);
    // Initialize REncoder.
    bl_size <<= 7;
    memory = (uint64_t)TOTAL_ITEM_NUM * BPK - bl_size;
    rencoder2.init(memory, HASH_NUM, 64, STORED_LEVELS, START_LEVEL, END_LEVEL);

    cout << "RBF memory  " << memory << endl;
    cout << "Insert " << TOTAL_ITEM_NUM << " items" << endl;
    cout << "Hash Num: " << HASH_NUM << endl;
    cout << "Bits per Key: " << BPK << endl;
    cout << "Begin level: " << START_LEVEL << endl;
    cout << "End level: " << END_LEVEL << endl;

    // Insert keys into REncoder.
    if (IS_SELF_ADAPT)
    {
        // Self-adaptively choose the optimal number of stored levels.
        int true_level = rencoder2.Insert_SelfAdapt(keys, SELF_ADAPT_STEP);
    }
    else
    {
        // Choose the number of stored levels given by arguments.
        if (END_LEVEL == -1)
        {
            rencoder2.stored_level = rencoder2.start_level + STORED_LEVELS;
            for (int i = 0; i < TOTAL_ITEM_NUM; i++)
            {
                rencoder2.Insert(keys[i], START_LEVEL);
            }
            rencoder2.stored_level -= 1;
        }
        else
        {
            rencoder2.start_level = max(1, END_LEVEL - STORED_LEVELS + 1);
            rencoder2.stored_level += 1;
            for (int i = 0; i < TOTAL_ITEM_NUM; i++)
            {
                rencoder2.Insert(keys[i], rencoder2.start_level);
            }
            rencoder2.stored_level -= 1;
        }
    }
    cout << "True begin level: " << rencoder2.start_level << endl;
    cout << "True end level: " << rencoder2.stored_level << endl;

    // Get the size of REncoder.
    pair<uint8_t *, size_t> ser = rencoder2.serialize();
    printf("%sREncoder size: %.2lf MB\n%s", kGreen, ser.second / 1.0 / 1024 / 1024, kWhite);
     for (int k = 0; k < repeat; k++) { 
        for (uint32_t i = 0; i < RANGE_QUERY_NUM; i++)
        { 
            uint64_t l = test_queries[k][i].first, r = test_queries[k][i].second; 
            bool res = rencoder2.RangeQuery(l, r); 
            auto iter = key_set.lower_bound(l);
            if (iter != key_set.end() && (*iter) <= r)
            {
                // Test for false negatives.
                if (!res)
                {
                    cout << "Range Query Error (False Negative)";
                    exit(-1);
                } 
                continue;
            } 
                    
        } 
    }
    // Get the false positive rate of REncoder.
    double FP = 0, TOTAL = 0;
    // Get the throughput of REncoder.
    cache_hit = 0;
    query_count = 0;
   // rencoder.black_list_hit = 0;
    long long TEST_QUERY_NUM = 0; 
    uint64_t duration = 0; 
    auto allstart = chrono::high_resolution_clock::now();
    for (int k = 0; k < repeat; k++) {
        
        auto start = chrono::high_resolution_clock::now();
        uint64_t cur_duration = 0;
        for (uint32_t i = 0; i < RANGE_QUERY_NUM; i++)
        {
            uint64_t l = test_queries[k][i].first, r = test_queries[k][i].second;
            TEST_QUERY_NUM++;
            bool res = rencoder2.RangeQuery(l, r); 
              
             if (!res_queries[k][i]) {
                TOTAL++;
                if (res)
                    {
                        FP++;  
                    rencoder2.add_black_list_node(l,r);
                    } 
            } 
                    
        }
        auto end = chrono::high_resolution_clock::now();
        cur_duration += chrono::duration_cast<chrono::nanoseconds>(end - start).count();
        printf("%sFalse Positive Rate: %lf\n%s", kGreen, FP / TOTAL, kWhite);
      //  printf("black list size : %lu\n",rencoder2.query_black_list_size());
        double throughput = (double)1000.0 * RANGE_QUERY_NUM / cur_duration;
        printf("%sFilter Throughput: %lf Mops/s\n%s", kGreen, throughput, kWhite);
         
    }
    auto allend = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::nanoseconds>(allend - allstart).count();
    double throughput = (double)1000.0 * TEST_QUERY_NUM / duration; 
        printf("%stotal Filter Throughput: %lf Mops/s\n%s", kGreen, throughput, kWhite);
    printf("query cnt: %lld; cache hit: %lld; hit rate: %lf\n", query_count / repeat, cache_hit / repeat, cache_hit / 1.0 / query_count);
    printf("black list size : %lu\n",rencoder2.query_black_list_size());
     
}

int main(int argc, char *argv[])
{
    if (argc < 8)
    {
        cout << "error arg : <KEY_TYPE> <TOTAL_ITEM_NUM> <RANGE_QUERY_NUM> <BPK> <IS_SELF_ADAPT> <SELF_ADAPT_STEP/STORED_LEVELS> <VERSION>" << endl;
        return 0;
    }

    KEY_TYPE = argv[1];              // Type of keys.
    TOTAL_ITEM_NUM = atoi(argv[2]);  // Number of keys.
    RANGE_QUERY_NUM = atoi(argv[3]); // Number of range queries.
    BPK = atof(argv[4]);             // Number of bits allocated for each key in REncoder.
    IS_SELF_ADAPT = atoi(argv[5]);   // Whether to insert keys self-adaptively.
    if (IS_SELF_ADAPT)
    {
        SELF_ADAPT_STEP = atoi(argv[6]); // Number of prefixes inserted for each key in each round.
    }
    else
    {
        STORED_LEVELS = atoi(argv[6]); // Number of stored levels given manually.
    }
    VERSION = argv[7]; // Version of REncoder.
    if (argc > 8) {
        SAME_RATE = atof(argv[8]);
    }
    else SAME_RATE = 1;
    if (version_set.find(VERSION) == version_set.end()) 
    {
        cout << "error version : REncoder/REncoderSS/REncoderSE" << endl;
        return 0;
    }
    printf("%sVersion: %s same rate : %lf\n%s", kGreen, VERSION.c_str(),SAME_RATE, kWhite);
    keys.resize(TOTAL_ITEM_NUM);
    range_queries.resize(RANGE_QUERY_NUM); 

    RunWorkload();

    return 0;
}