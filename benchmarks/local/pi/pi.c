// This code was based on
// http://software.intel.com/en-us/articles/writing-parallel-programs-a-multi-language-tutorial-introduction/

#include "omp.h"
#include "stdio.h"
#include "hooks_base.h"
#include <unistd.h>

long num_steps;
double step;

void main(int argc, char* argv[])
{
   int i;
   double x[1000];
   
   num_steps = 500;

   parmacs_roi_begin();

   #pragma omp parallel for private(x)
   for (i=0;i<= num_steps; i++)
   {
      for (int j=0; j < 50; j++)
      {
      x[j] = j*i/1.2;
      }
      //usleep(20);

      for (int j=0; j < 30; j++)
      {
      x[j] = i;
      }
      //usleep(20);

      for (int j=0; j < 50; j++)
      {
      x[j] = j*i/1.2 + j*j/2.1;
      }
      //usleep(20);

      for (int j=0; j < 70; j++)
      {
      x[j] = i;
      }
      //usleep(20);


   }

   parmacs_roi_end();

}
