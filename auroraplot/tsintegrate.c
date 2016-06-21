
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

/******************************************************************************
*    RESAMPLING AND INTEGRATION OF TIME SERIES DATA                           *
******************************************************************************/
struct OutStruct {
    double  *data;
    double  *weight;
} *output;


struct OutStruct *tsIntegrate(int64_t nData, int64_t *dataST, int64_t *dataET,
                              double *data, double *weight,
                              int64_t nReq, int64_t *reqST, int64_t *reqET) {

    // local stack variables
    int64_t iData, iReq, iDataLowLim, iDataUpLim;
    int64_t tempST, tempET;
    double tempData, tempWeight;    

    // Allocate memory from the heap
    output = (struct OutStruct *)malloc(sizeof(struct OutStruct));
    output->data = (double *)malloc(sizeof(double)*nReq);
    output->weight = (double *)malloc(sizeof(double)*nReq);
     
    // Find lower/upper limit of required data index to speed up the loop
    // when there is a large data array, with many samples outside of the 
    // requested range.
    iDataLowLim=0;
    iDataUpLim=nData;
    for (iData=0;iData<nData;iData++) { 
        if (reqST[0]>=dataET[iData]) {
            iDataLowLim = iData;
        };
        if (reqET[nReq-1]<=dataST[iData]) {
            iDataUpLim = iData;
            break;
        };
    };
     
    for (iReq=0; iReq<nReq; iReq++) {
        tempData = 0.0;
        tempWeight = 0.0;
        output->data[iReq] = 0.0;
        output->weight[iReq] = 0.0;
        for (iData=iDataLowLim; iData<iDataUpLim; iData++) { 
            if (reqST[iReq]>=dataET[iData]) {
                /* If requested data start times, reqST, are sorted,
                   then there are no usefull samples before iData
                   for any subsequent requested samples. We can 
                   set the new lower limit of data sample loop to 
                   the current iData value.                            */
                iDataLowLim = iData;
                continue;
            };
            if (reqET[iReq]<=dataST[iData]) {
                /* If dataST are sorted, this means that there are no
                   more data samples that will be needed in this loop.
                   We can move on to the next requested sample.        */
                break;
            };
            /* If the program gets here, there is overlap between
               the current data sample and the requested sample.
               tempST and tempET will define the start and end points
               of this overlap.                                        */
            if (reqST[iReq]<dataST[iData]) {
                tempST = dataST[iData];
            } else { 
                tempST = reqST[iReq];
            };
            if (reqET[iReq]>dataET[iData]) {
                tempET = dataET[iData];
            } else { 
                tempET = reqET[iReq];
            };
            /* Note that the new data values are not divided by the
               new weights. That can be done at a later stage if 
               necessary.                                              */
            tempData += data[iData] * weight[iData] * (double)(tempET-tempST)
                                      / (double)(dataET[iData]-dataST[iData]);
            tempWeight += weight[iData] * (double)(tempET-tempST)
                          / (double)(dataET[iData]-dataST[iData]);
        };
        output->data[iReq] = tempData;
        output->weight[iReq] = tempWeight;
    };
    return output;
};


void freemem() {
    free(output->data);
    free(output->weight);
    free(output);
};
