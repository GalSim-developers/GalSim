
#ifndef STOPWATCH_H
#define STOPWATCH_H

#include <sys/time.h>

namespace galsim {

class Stopwatch 
{
private:
    double seconds;
    struct timeval tpStart;
    bool running;
public:
    Stopwatch() : seconds(0.), running(false) {}

    void start() { gettimeofday(&tpStart, NULL); running=true; }

    void stop() 
    {
        if (!running) return;
        struct timeval tp;
        gettimeofday(&tp, NULL);
        seconds += (tp.tv_sec - tpStart.tv_sec)
            + 1e-6*(tp.tv_usec - tpStart.tv_usec);
        running = false;
    }
    void reset() { seconds=0.; running=false; }
    operator double() const { return seconds; }
};

}

#endif
