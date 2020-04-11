/**
 * @file GraphColor.cc
 * @author  Songjie Niu, Shimin Chen
 * @version 0.1
 *
 * @section LICENSE 
 * 
 * Copyright 2016 Shimin Chen (chensm@ict.ac.cn) and
 * Songjie Niu (niusongjie@ict.ac.cn)
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 * @section DESCRIPTION
 * 
 * This file implements the PageRank algorithm using graphlite API.
 *
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <ctime>
#include <set>
#include <map>

#include "GraphLite.h"

#define VERTEX_CLASS_NAME(name) GraphColor##name

#define EPS 1e-6

int64_t startVertex=0;
int colorNums=0;

class VERTEX_CLASS_NAME(InputFormatter): public InputFormatter {
public:
    int64_t getVertexNum() {
        unsigned long long n;
        sscanf(m_ptotal_vertex_line, "%lld", &n);
        m_total_vertex= n;
        return m_total_vertex;
    }
    int64_t getEdgeNum() {
        unsigned long long n;
        sscanf(m_ptotal_edge_line, "%lld", &n);
        m_total_edge= n;
        return m_total_edge;
    }
    int getVertexValueSize() {
        m_n_value_size = sizeof(int);
        return m_n_value_size;
    }
    int getEdgeValueSize() {
        m_e_value_size = sizeof(int);
        return m_e_value_size;
    }
    int getMessageValueSize() {
        m_m_value_size = sizeof(int);
        return m_m_value_size;
    }
    void loadGraph() {
        unsigned long long last_vertex;
        unsigned long long from;
        unsigned long long to;
        int weight = 0; //do not use weight, change type to int
        
        int defaulColor = -1; //set default color
        int outdegree = 0;
        
        const char *line= getEdgeLine();

        // Note: modify this if an edge weight is to be read
        //       modify the 'weight' variable

        // printf("start in %d \n",int(colorNums));

        sscanf(line, "%lld %lld", &from, &to);
        addEdge(from, to, &weight);

        last_vertex = from;
        ++outdegree;
        for (int64_t i = 1; i < m_total_edge; ++i) {
            line= getEdgeLine();

            // Note: modify this if an edge weight is to be read
            //       modify the 'weight' variable

            sscanf(line, "%lld %lld", &from, &to);
            if (last_vertex != from) {
                addVertex(last_vertex, &defaulColor, outdegree);
                last_vertex = from;
                outdegree = 1;
            } else {
                ++outdegree;
            }
            addEdge(from, to, &weight);
        }
        addVertex(last_vertex, &defaulColor, outdegree);
    }
};

class VERTEX_CLASS_NAME(OutputFormatter): public OutputFormatter {
public:
    void writeResult() {
        // printf("start in %d \n",int(colorNums));
        int64_t vid;
        int value;
        char s[1024];

        for (ResultIterator r_iter; ! r_iter.done(); r_iter.next() ) {
            r_iter.getIdValue(vid, &value);
            int n = sprintf(s, "%lld: %d\n", (unsigned long long)vid, value);
            writeNextResLine(s, n);
        }
    }
};

// An aggregator that records a double value tom compute sum
// graph coloring did not use Aggregator
class VERTEX_CLASS_NAME(Aggregator): public Aggregator<int> {
public:
    void init() { }
    void* getGlobal() {
        return &m_global;
    }
    void setGlobal(const void* p) {
        m_global = * (int *)p;
    }
    void* getLocal() {
        return &m_local;
    }
    void merge(const void* p) { }
    void accumulate(const void* p) { }
};

// graph coloring 
class VERTEX_CLASS_NAME(): public Vertex <int, int, int> {
public:
    void compute(MessageIterator* pmsgs) {
        int val;
        int64_t vertId = getVertexId();
        int vertColor = *mutableValue();

        srand(time(0) * vertId * clock()); 
        if (getSuperstep() == 0) {
            // init all vertex's color 
            if(vertId == startVertex){
                val = 0;
            }else{
                val = rand() % colorNums;
            }
        } else {
            // printf("step: %d , vertId: %d, vertColor: %d\n",int(getSuperstep()), val, int(vertId), vertColor);
            set<int> neighborClors; 
            // int countNeighbors = 0;
            // get all neighbor's color
            for ( ; ! pmsgs->done(); pmsgs->next() ) {
                // ++countNeighbors;
                int nbc = pmsgs->getValue();
                if(nbc != -1){
                    neighborClors.insert(nbc);
                }
            }
            int nSize = neighborClors.size();
            // printf("\n neighb size: %d\n", int(nSize));
            // printf("\n neighb num: %d\n", int(countNeighbors));

            val = vertColor;
            if(val == -1){
                while(true){
                    val = rand() % colorNums;
                    if(neighborClors.find(val)==neighborClors.end()){
                        break;
                    }
                }
            }else{
                if(neighborClors.find(val) == neighborClors.end()){
                    voteToHalt(); return ;
                }else{
                    while(true){
                        val = rand() % colorNums;
                        if(neighborClors.find(val)==neighborClors.end()){
                            break;
                        }
                    }
                }
            }
        }
        * mutableValue() = val;
        sendMessageToAllNeighbors(val);
    }
};

class VERTEX_CLASS_NAME(Graph): public Graph {
public:
    VERTEX_CLASS_NAME(Aggregator)* aggregator;

public:
    // argv[0]: GraphColor.so
    // argv[1]: <input path>
    // argv[2]: <output path>
    // argv[3]: <v0 id> 
    // argv[4]: <num color>
    void init(int argc, char* argv[]) {

        setNumHosts(5);
        setHost(0, "localhost", 1411);
        setHost(1, "localhost", 1421);
        setHost(2, "localhost", 1431);
        setHost(3, "localhost", 1441);
        setHost(4, "localhost", 1451);

        if (argc < 5) {
           printf ("Usage: %s <input path> <output path> <v0 id> <num color>\n", argv[0]);
           exit(1);
        }

        m_pin_path = argv[1];
        m_pout_path = argv[2];

        startVertex = atoll(argv[3]);
		colorNums = atoi(argv[4]);
        // printf("%d \n",int(startVertex));
        // printf("%d \n",int(colorNums));

        aggregator = new VERTEX_CLASS_NAME(Aggregator)[1];
        regNumAggr(1);
        regAggr(0, &aggregator[0]);
    }
    void term() {
        delete[] aggregator;
    }
};

/* STOP: do not change the code below. */
extern "C" Graph* create_graph() {
    Graph* pgraph = new VERTEX_CLASS_NAME(Graph);
    

    pgraph->m_pin_formatter = new VERTEX_CLASS_NAME(InputFormatter);
    pgraph->m_pout_formatter = new VERTEX_CLASS_NAME(OutputFormatter);
    pgraph->m_pver_base = new VERTEX_CLASS_NAME();

    return pgraph;
}

extern "C" void destroy_graph(Graph* pobject) {
    delete ( VERTEX_CLASS_NAME()* )(pobject->m_pver_base);
    delete ( VERTEX_CLASS_NAME(OutputFormatter)* )(pobject->m_pout_formatter);
    delete ( VERTEX_CLASS_NAME(InputFormatter)* )(pobject->m_pin_formatter);
    delete ( VERTEX_CLASS_NAME(Graph)* )pobject;
}
