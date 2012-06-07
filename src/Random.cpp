
#include <sys/time.h>
#include "Random.h"

namespace galsim {

    void BaseDeviate::seedtime() 
    {
        struct timeval tp;
        gettimeofday(&tp,NULL);
        _rng->seed(tp.tv_usec);
    }

}
