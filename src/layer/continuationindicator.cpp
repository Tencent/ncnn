//
// Addition by Jolibrain
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "continuationindicator.h"
#include <iostream>

namespace ncnn {

  DEFINE_LAYER_CREATOR(ContinuationIndicator)
  
  ContinuationIndicator::ContinuationIndicator()
  {
    one_blob_only = true;
    support_inplace = false;
    support_vulkan = false;
  }

  int ContinuationIndicator::load_param(const ParamDict &pd)
  {
    time_step = pd.get(0, 0);
    axis = pd.get(1, 0);
    return 0;
  }

  int ContinuationIndicator::forward(const Mat &bottom_blob, Mat &top_blob, const Option &opt) const
  {
    size_t elemsize = bottom_blob.elemsize;
    top_blob.create(1, time_step, elemsize, opt.blob_allocator);
    float *tbc = top_blob.channel(0);
    
    for (int t=0;t<time_step;++t)
      {
	for (size_t b=0;b<elemsize;++b)
	  {
	    *tbc++ = t == 0 ? 0.0f : 1.0f;
	  }
      }
    return 0;
  }
  
}
