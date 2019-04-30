/**
 * @File   : objection_detection.cpp
 * @Author : damone (damonexw@gmail.com)
 * @Link   : 
 * @Date   : 11/2/2018, 2:50:06 PM
 */

#include "objectdetection.h"

namespace ncnn
{

float box_overlap(float min1, float max1, float min2, float max2)
{
    float left = min1 > min2 ? min1 : min2;
    float right = max1 < max2 ? max1 : max2;
    return right - left;
}

float ObjectBox::box_area() const
{
    return (ymax - ymin) * (xmax - xmin);
}

float ObjectBox::box_intersection(const ObjectBox &other) const
{
    float intersection_w = box_overlap(xmin, xmax, other.xmin, other.xmax);
    float intersection_h = box_overlap(ymin, ymax, other.ymin, other.ymax);
    if (intersection_w < 0 || intersection_h < 0)
        return 0;
    float area = intersection_w * intersection_h;
    return area;
}

float ObjectBox::box_union(const ObjectBox &other) const
{
    float intersection_s = box_intersection(other);
    float u = box_area() + other.box_area() - intersection_s;
    return u;
}

float ObjectBox::box_iou(const ObjectBox &other) const
{
    return box_intersection(other) / (box_union(other));
}

ObjectsManager::ObjectsManager(size_t steps) : object_boxs_table()
{
    step = prob_steps_MAX / steps;
    step = step <= 0 ? 1 : step;
    prob_steps = prob_steps_MAX / step;
    for (int i = 0; i < prob_steps; i++)
    {
        PObjectBoxs pboxs = (new TObjectBoxs(object_compare));
        object_boxs_table.push_back(pboxs);
    }
}

ObjectsManager::~ObjectsManager()
{
    for (int i = 0; i < prob_steps; i++)
    {
        delete object_boxs_table[i];
        object_boxs_table[i] = NULL;
    }
    object_boxs_table.clear();
}

void ObjectsManager::add_new_object_box(const ObjectBox &object_box)
{
    int index = get_prob_index(object_box.prob);
    object_boxs_table[index]->insert(object_box);
}

void ObjectsManager::do_objects_nms(std::vector<ObjectBox> &objects, float nms, float prob_threh)
{
    objects.clear();
    int start = prob_steps - 1;
    int min_index = get_prob_index(prob_threh);
    for (; start >= min_index; start--)
    {
        TObjectBoxs &object_boxs = *object_boxs_table[start];
        if (object_boxs.empty())
        {
            continue;
        }

        do
        {
            ObjectBox &object_box = object_boxs.front();
            objects.push_back(object_box);
            
            ObjectBoxNmsCondition condition(object_box, nms);
            object_boxs.pop_front();
            
            object_boxs.remove_all_match(condition);
            
            do_object_boxs_table_nms(condition, min_index, start);

        } while (!object_boxs.empty());
    }
}

int ObjectsManager::object_compare(const ObjectBox &box1, const ObjectBox &box2)
{
    return box1.prob > box2.prob ? 1 : -1;
}

inline void ObjectsManager::do_object_boxs_table_nms(ObjectBoxNmsCondition &condition, int start_index,  int end_index)
{
    for (int i = start_index; i < end_index; i++)
    {
        TObjectBoxs &object_boxs = *object_boxs_table[i];
        if (object_boxs.empty())
        {
            continue;
        }

        object_boxs.remove_all_match(condition);
    }
}

inline int ObjectsManager::get_prob_index(float prob)
{
    int index = (int)(prob * 100) / step;
    index = index < 0 ? 0 : index;
    index = index < prob_steps ? index : prob_steps - 1;
    return index;
}

} // namespace ncnn
