/*
 * PrimalSolver - a convex optimization solver in C99 (LP/QP/SOCP/SDP/exp-power/MIP).
 * Copyright 2026 Gaetano Minardi
 * SPDX-License-Identifier: Apache-2.0
 * 
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License.  A copy of the License
 * is in the repository root (LICENSE) and at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

#include "conic_matrix.h"
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

ConicMatrix *conic_matrix_new(int rows,int cols) {
    if(rows<0||cols<0||(size_t)rows>SIZE_MAX/sizeof(ConicRow))return NULL;
    ConicMatrix *m=calloc(1,sizeof(*m));
    if(!m)return NULL;
    m->row=calloc((size_t)(rows?rows:1),sizeof(*m->row));
    if(!m->row){free(m);return NULL;}
    m->rows=rows;m->cols=cols;
    m->bytes=sizeof(*m)+(size_t)(rows?rows:1)*sizeof(*m->row);
    return m;
}
void conic_matrix_free(ConicMatrix *m) {
    if(!m)return;
    for(int i=0;i<m->used;i++)free(m->row[i].entry);
    free(m->row);free(m);
}
static void put(ConicMatrix *m,int row,int col,double value,int add) {
    if(!m||m->failed)return;
    if(row<0||row>=m->rows||col<0||col>=m->cols){m->failed=1;return;}
    ConicRow *r=m->row+row;
    if(col<=r->maxcol)for(int i=0;i<r->nz;i++)if(r->entry[i].col==col) {
        double v=add?r->entry[i].value+value:value;
        if(v==0)r->entry[i]=r->entry[--r->nz];
        else r->entry[i].value=v;
        return;
    }
    if(value==0)return;
    if(r->nz==r->cap) {
        if(r->cap>INT_MAX/2){m->failed=1;return;}
        int cap=r->cap?r->cap*2:4;
        if((size_t)cap>SIZE_MAX/sizeof(*r->entry)){m->failed=1;return;}
        ConicEntry *a=realloc(r->entry,(size_t)cap*sizeof(*a));
        if(!a){m->failed=1;return;}
        m->bytes+=(size_t)(cap-r->cap)*sizeof(*a);r->cap=cap;r->entry=a;
    }
    r->entry[r->nz].col=col;r->entry[r->nz++].value=value;
    if(col>r->maxcol)r->maxcol=col;
    if(row>=m->used)m->used=row+1;
}
void conic_matrix_set(ConicMatrix *m,int r,int c,double v){put(m,r,c,v,0);}
void conic_matrix_add(ConicMatrix *m,int r,int c,double v){put(m,r,c,v,1);}
void conic_matrix_clear(ConicMatrix *m,int first) {
    if(!m||first<0){if(m)m->failed=1;return;}
    for(int i=first;i<m->used;i++){m->row[i].nz=0;m->row[i].maxcol=0;}
}
static int entry_order(const void *a,const void *b) {
    int x=((const ConicEntry*)a)->col,y=((const ConicEntry*)b)->col;
    return (x>y)-(x<y);
}
void conic_matrix_sort(ConicMatrix *m) {
    if(m)for(int i=0;i<m->used;i++)if(m->row[i].nz>1)
        qsort(m->row[i].entry,(size_t)m->row[i].nz,sizeof(ConicEntry),entry_order);
}
void conic_csr_free(SocpCsr *m) {
    if(!m)return;
    free(m->ptr);free(m->idx);free(m->val);memset(m,0,sizeof(*m));
}
int conic_matrix_csr(ConicMatrix *m,int rows,SocpCsr *out) {
    memset(out,0,sizeof(*out));
    if(!m||m->failed||rows<0||rows>m->rows)return -1;
    int nz=0;
    for(int i=0;i<rows;i++) {
        if(m->row[i].nz>INT_MAX-nz)return -1;
        nz+=m->row[i].nz;
    }
    out->ptr=calloc((size_t)rows+1,sizeof(int));
    out->idx=malloc((size_t)(nz?nz:1)*sizeof(int));
    out->val=malloc((size_t)(nz?nz:1)*sizeof(double));
    if(!out->ptr||!out->idx||!out->val){conic_csr_free(out);return -1;}
    conic_matrix_sort(m);int q=0;
    for(int i=0;i<rows;i++) {
        const ConicRow *r=m->row+i;
        for(int j=0;j<r->nz;j++){out->idx[q]=r->entry[j].col;out->val[q++]=r->entry[j].value;}
        out->ptr[i+1]=q;
    }
    out->rows=rows;out->cols=m->cols;return 0;
}
double *conic_matrix_dense(const ConicMatrix *m,int rows) {
    if(!m||m->failed||rows<0||rows>m->rows)return NULL;
    if(m->cols&&(size_t)rows>SIZE_MAX/sizeof(double)/(size_t)m->cols)return NULL;
    size_t count=(size_t)rows*m->cols;
    double *a=calloc(count?count:1,sizeof(double));
    if(a)for(int i=0;i<rows;i++)for(int j=0;j<m->row[i].nz;j++)
        a[(size_t)i*m->cols+m->row[i].entry[j].col]=m->row[i].entry[j].value;
    return a;
}
