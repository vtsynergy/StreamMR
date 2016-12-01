#include "rdtsc.h"

cl_event ocdTempEvent;
#ifdef ENABLE_TIMER
cl_ulong startTime, endTime;

struct ocdTimer * ocdTempTimer;
struct ocdDualTimer * ocdTempDualTimer;
struct ocdHostTimer * ocdTempHostTimer;
struct ocdHostTimer fullExecTimer = {OCD_TIMER_HOST, NULL, 0, 0, 0, {0, 0}};

struct timer_group_mem head = {NULL, NULL, NULL};
struct timer_group_mem * tail;

char rootStr[1] = { (char)0};
cl_ulong rootTimes[7] = {0, 0, 0, 0, 0, 0, 0};
cl_ulong totalTimes[7] = {0, 0, 0, 0, 0, 0, 0};

struct timer_name_tree_node  root = {
    rootStr, 0, NULL, NULL, &head, 0, rootTimes
}; //sentinel

//linear search of the Name List.
//returns a pointer to the correct time array, or -1 if none exists yet
//rather inefficient if many names are used, but the tree will take care of
// speeding lookups, and we'll switch to alpha sort by default as a sideffect
void * checkSimpleNameList(const char * s, int len) {
    struct timer_name_tree_node * curr = root.next;
    while (curr != NULL) { //still unique names to be checked
        if (strcmp(s, curr->string) == 0) {
            return curr->times;
        }
        curr = curr->next;
    }
    return (void *)-1;
}

struct timer_name_tree_node * atail = &root;
//simple named timer aggregation
//linear scan of the timer list, adds nodes to a names list as necessary
//DO NOT USE AT THE SAME TIME AS THE TREE
//this replaces the tree with a simple unordered list
void simpleNameTally() {
    struct timer_group_mem * curr = head.next;
    while (curr != NULL) {
        void * time;
        if (curr->timer->s.name != NULL) {
            time = checkSimpleNameList(curr->timer->s.name, curr->timer->s.nlen);
            if (time == (void *)-1) {
                //initialize a new name list node
                atail->next = (struct timer_name_tree_node *) calloc(sizeof (struct timer_name_tree_node), 1);
                atail = atail->next;
                atail->next = NULL;
                atail->len = curr->timer->s.nlen;
                atail->string = curr->timer->s.name;
                atail->times = (cl_ulong *) calloc(sizeof(cl_ulong), 7);
                time = (void *)atail->times;
            }} else {
            time = (void *)root.times;
            }
            if (curr->timer->s.endtime > curr->timer->s.starttime) {
                switch (curr->timer->s.type) {
                    case OCD_TIMER_D2H:
                ((cl_ulong *) time)[1] += curr->timer->s.endtime - curr->timer->s.starttime;
                totalTimes[1] +=curr->timer->s.endtime - curr->timer->s.starttime;
                break;
                
                    case OCD_TIMER_H2D:
                ((cl_ulong *) time)[2] += curr->timer->s.endtime - curr->timer->s.starttime;
                totalTimes[2] +=curr->timer->s.endtime - curr->timer->s.starttime;
                break;
                                        
                    case OCD_TIMER_D2D:
                ((cl_ulong *) time)[3] += curr->timer->s.endtime - curr->timer->s.starttime;
                totalTimes[3] +=curr->timer->s.endtime - curr->timer->s.starttime;
                break;
                        
                    case OCD_TIMER_KERNEL:
                ((cl_ulong *) time)[4] += curr->timer->s.endtime - curr->timer->s.starttime;
                totalTimes[4] +=curr->timer->s.endtime - curr->timer->s.starttime;
                break;
                        
                    case OCD_TIMER_HOST:
                ((cl_ulong *) time)[5] += curr->timer->s.endtime - curr->timer->s.starttime;
                totalTimes[5] +=curr->timer->s.endtime - curr->timer->s.starttime;
                break;
                        
                    case OCD_TIMER_DUAL:
                ((cl_ulong *) time)[6] += curr->timer->s.endtime - curr->timer->s.starttime;
                totalTimes[6] +=curr->timer->s.endtime - curr->timer->s.starttime;
                break;
                }
                ((cl_ulong *) time)[0] += curr->timer->s.endtime - curr->timer->s.starttime;
            
                }
        
        curr = curr->next;
    }
    totalTimes[0] = fullExecTimer.endtime - fullExecTimer.starttime;

}


//assumes simpleNameTally was already called (once) to add up timers
//now culls off zero-value timers
void simpleNamePrint() {
    struct timer_name_tree_node * curr = &root;
    while (curr != NULL) { //still unique names to be checked
        if (curr->times[0] > 0) {if (strcmp(curr->string, rootStr) != 0) {// if the string isn't empty
        printf("Timer [%s]: \t %llu\n", curr->string, curr->times[0]);
        } else {
        printf("Unnamed Timers: \t %llu\n", curr->times[0]);
        }
        if (curr->times[1] > 0) printf("\tD2H:    \t %llu\n", curr->times[1]);
        if (curr->times[2] > 0) printf("\tH2D:    \t %llu\n", curr->times[2]);
        if (curr->times[3] > 0) printf("\tD2D:    \t %llu\n", curr->times[3]);
        if (curr->times[4] > 0) printf("\tKernel: \t %llu\n", curr->times[4]);
        if (curr->times[5] > 0) printf("\tHost:   \t %llu\n", curr->times[5]);
        if (curr->times[6] > 0) printf("\tDual:   \t %llu\n", curr->times[6]);
        }
        curr = curr->next;
    }
}

//chews up the timer list from head to tail, deallocating all nodes
void destTimerList() {
    struct timer_group_mem * temp, * curr = head.next;
    temp = curr;
    //make sure we can't try to do another cleanup
    head.next = NULL;
    while (curr != NULL) {
        //slide the window
        curr = curr->next;
        //cleanup behind
        if (temp != NULL) {
            if (temp->timer !=NULL) free(temp->timer);
            free(temp);
        }
        //catch up
        temp = curr;
    }
}

//chews up the simpleNameList from root to atail, deallocating all nodes
void destNameList() {
    struct timer_name_tree_node * temp, * curr = root.next;
    temp = curr;
    //make sure we can't try to do another cleanup
    root.next = NULL;
    while (curr != NULL) {
        curr=curr->next;
        if (temp !=NULL) {
            if (temp->times != NULL) free(temp->times);
            free(temp);
        }
        temp = curr;
    }
}

void * getTimePtr(cl_event e) {
    struct timer_group_mem * curr = head.next;
    while (curr != 0) {
        //if composed, will be a negative value
        if (curr->timer->s.type > 0) {
            if (curr->timer->s.event == e) return (void *) curr->timer;
        }
        curr = curr->next;
    }
    return (void *) - 1;
}
//only returns a composed timer with events matching both e1 and e2, in either order

void * getDualTimePtr(cl_event e1, cl_event e2) {
    struct timer_group_mem * curr = head.next;
    while (curr != 0) {
        //if composed, will be a negative value
        if (curr->timer->s.type < 0) {
            if ((curr->timer->c.event[0] == e1 \
                    && curr->timer->c.event[1] == e2) \
                    || (curr->timer->c.event[0] == e2 \
                    && curr->timer->c.event[1] == e1)) \
                return (void *) curr->timer;
        }
        curr = curr->next;
    }
    return (void *) - 1;
}

//simply adds timer t to the end of the list

void addTimer(union ocdInternalTimer * t) {
    if (head.next == NULL) { //no members
        tail = &head; //reset tail, just incase
    } else {
        while (tail->next != NULL) { //slide tail to the end
            tail = tail->next;
        }
    }
    struct timer_group_mem * temp_wrap = (struct timer_group_mem *) calloc(sizeof (struct timer_group_mem), 1);

    temp_wrap->next = NULL;
    temp_wrap->timer = t;
    tail->next = temp_wrap;
    tail = temp_wrap;
}

//irreversible! Only call immediately before freeing the timer!

int removeTimer(union ocdInternalTimer * t) {
    struct timer_group_mem * curr = head.next, * old = &head;
    while (curr != 0 && curr->timer != t) {
        old = curr;
        curr = curr->next;
    }
    if (curr != 0) {
        if (curr->next == 0) { //we are the tail!
            tail = old; //so back the tail up one
        }
        old->next = curr->next;
        free(curr);
        return 0;
    }
    return -1; // probably should free(groups) somehow 
}


#ifdef TIMER_TEST
//Debug call for checking list construction

void walkList() {
    struct timer_group_mem * curr = head.next;
    fprintf(stderr, "Walking list starting at [%lx]--[%lx]\n", (unsigned long) &head, (unsigned long) head.timer);
    while (curr != 0) {
        fprintf(stderr, "\t[%lx]--[%lx]\n", (unsigned long) curr, (unsigned long) curr->timer->s.event);
        curr = curr->next;
    }
}

#endif //TIMER_TEST

#endif //ENABLE_TIMER

cl_device_id
GetDevice(int platform, int device) {
    cl_int err;
    cl_uint nPlatforms = 1;
    err = clGetPlatformIDs(0, NULL, &nPlatforms);
    CHECK_ERROR(err);

    if (nPlatforms <= 0) {
        printf("No OpenCL platforms found. Exiting.\n");
        exit(0);
    }
    if (platform < 0 || platform >= nPlatforms) // platform ID out of range
    {
        printf("Platform index %d is out of range. \n", platform);
        exit(-4);
    }
    cl_platform_id *platforms = (cl_platform_id *) malloc(sizeof (cl_platform_id) * nPlatforms);
    err = clGetPlatformIDs(nPlatforms, platforms, NULL);
    CHECK_ERROR(err);

    cl_uint nDevices = 1;
    char platformName[100];
    err = clGetPlatformInfo(platforms[0], CL_PLATFORM_VENDOR, sizeof (platformName), platformName, NULL);
    CHECK_ERROR(err);
    printf("Platform Chosen : %s\n", platformName);
    // query devices
    err = clGetDeviceIDs(platforms[platform], USEGPU ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 0, NULL, &nDevices);
    CHECK_ERROR(err);
    if (nDevices <= 0) {
        printf("No OpenCL Device found. Exiting.\n");
        exit(0);
    }
    if (device < 0 || device >= nDevices) // platform ID out of range
    {
        printf("Device index %d is out of range. \n", device);
        exit(-4);
    }
    cl_device_id* devices = (cl_device_id *) malloc(sizeof (cl_device_id) * nDevices);
    err = clGetDeviceIDs(platforms[platform], USEGPU ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, nDevices, devices, NULL);
    CHECK_ERROR(err);
    char DeviceName[100];
    err = clGetDeviceInfo(devices[device], CL_DEVICE_NAME, sizeof (DeviceName), DeviceName, NULL);
    CHECK_ERROR(err);
    printf("Device Chosen : %s\n", DeviceName);

    return devices[device];
}

const char *get_error_string(cl_int err){
    switch(err){
        case 0: return "CL_SUCCESS";
        case -1: return "CL_DEVICE_NOT_FOUND";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        case -3: return "CL_COMPILER_NOT_AVAILABLE";
        case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5: return "CL_OUT_OF_RESOURCES";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -8: return "CL_MEM_COPY_OVERLAP";
        case -9: return "CL_IMAGE_FORMAT_MISMATCH";
        case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -11: return "CL_BUILD_PROGRAM_FAILURE";
        case -12: return "CL_MAP_FAILURE";
        case -30: return "CL_INVALID_VALUE";
        case -31: return "CL_INVALID_DEVICE_TYPE";
        case -32: return "CL_INVALID_PLATFORM";
        case -33: return "CL_INVALID_DEVICE";
        case -34: return "CL_INVALID_CONTEXT";
        case -35: return "CL_INVALID_QUEUE_PROPERTIES";
        case -36: return "CL_INVALID_COMMAND_QUEUE";
        case -37: return "CL_INVALID_HOST_PTR";
        case -38: return "CL_INVALID_MEM_OBJECT";
        case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case -40: return "CL_INVALID_IMAGE_SIZE";
        case -41: return "CL_INVALID_SAMPLER";
        case -42: return "CL_INVALID_BINARY";
        case -43: return "CL_INVALID_BUILD_OPTIONS";
        case -44: return "CL_INVALID_PROGRAM";
        case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46: return "CL_INVALID_KERNEL_NAME";
        case -47: return "CL_INVALID_KERNEL_DEFINITION";
        case -48: return "CL_INVALID_KERNEL";
        case -49: return "CL_INVALID_ARG_INDEX";
        case -50: return "CL_INVALID_ARG_VALUE";
        case -51: return "CL_INVALID_ARG_SIZE";
        case -52: return "CL_INVALID_KERNEL_ARGS";
        case -53: return "CL_INVALID_WORK_DIMENSION";
        case -54: return "CL_INVALID_WORK_GROUP_SIZE";
        case -55: return "CL_INVALID_WORK_ITEM_SIZE";
        case -56: return "CL_INVALID_GLOBAL_OFFSET";
        case -57: return "CL_INVALID_EVENT_WAIT_LIST";
        case -58: return "CL_INVALID_EVENT";
        case -59: return "CL_INVALID_OPERATION";
        case -60: return "CL_INVALID_GL_OBJECT";
        case -61: return "CL_INVALID_BUFFER_SIZE";
        case -62: return "CL_INVALID_MIP_LEVEL";
        case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
        default: return "Unknown OpenCL error";
    }
}
