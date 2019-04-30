/**
 * @File   : objection_detection.h
 * @Author : damone (damonexw@gmail.com)
 * @Link   : 
 * @Date   : 11/2/2018, 2:47:51 PM
 */

#ifndef _OBJECT_DETECTION_H
#define _OBJECT_DETECTION_H

#include "threadtools.h"
#include <memory>
#include <vector>
#include <stdio.h>

namespace ncnn
{

struct ObjectBox
{
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    float prob;
    int classid;

    float box_area() const;
    float box_intersection(const ObjectBox &other) const;
    float box_union(const ObjectBox &other) const;
    float box_iou(const ObjectBox &other) const;
};

struct ObjectBoxPrintOp
{
    bool operator()(ObjectBox &object)
    {
        printf("object_box: xin %f ymin %f xmax %f ymax %f prob: %f classid ：%d \n",
               object.xmin, object.ymin, object.xmax, object.ymax, object.prob,
               object.classid);
        return true;
    }
};

struct ObjectBoxNmsCondition
{
    ObjectBoxNmsCondition(ObjectBox &first, float nms) : top_box(first), nms_threshold(nms) {}

    bool operator()(const ObjectBox &object)
    {
        return top_box.classid == object.classid && top_box.prob > object.prob && top_box.box_iou(object) > nms_threshold;
    }

  private:
    ObjectBox top_box;
    float nms_threshold;
};

typedef ncnn::SafeOrderList<ObjectBox> TObjectBoxs;
typedef TObjectBoxs *PObjectBoxs;

class ObjectsManager
{
    enum ProbGrid
    {
        prob_steps_MAX = 100,
    };

  public:
    ObjectsManager(size_t steps);
    ~ObjectsManager();

    void add_new_object_box(const ObjectBox &object_box);
    void do_objects_nms(std::vector<ObjectBox> &objects, float nms, float prob_threh = 0.f);

  private:
    // disallow the copy constructor and operator= function
    ObjectsManager(const ObjectsManager &);
    void operator=(const ObjectsManager &);

  protected:
    static int object_compare(const ObjectBox &box1, const ObjectBox &box2);
    inline void do_object_boxs_table_nms(ObjectBoxNmsCondition &condition, int start_index, int end_index);
    inline int get_prob_index(float prob);

  protected:
    std::vector<PObjectBoxs> object_boxs_table;
    size_t prob_steps;
    size_t step;
};

} // namespace ncnn

#endif
