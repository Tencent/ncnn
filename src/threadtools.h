/**
 * @File   : thread_tools.h
 * @Author : damone (damonexw@gmail.com)
 * @Link   : 
 * @Date   : 11/2/2018, 3:26:56 PM
 */

#ifndef _TGUARD_H
#define _TGUARD_H

#include "allocator.h"
#include <list>

namespace ncnn
{

class TGuard
{
public:
  explicit TGuard(ncnn::Mutex &mutex) : crit(mutex) { crit.lock(); }
  ~TGuard() { crit.unlock(); }

private:
  ncnn::Mutex &crit;
};

template <typename T>
class SafeOrderList
{
public:
  typedef std::list<T> TContainer;
  typedef typename TContainer::iterator Iterator;

  /**
   * compare function
   * if first <  second return -1
   * if first >  second return 1
   * if first == second return 0
   */
  typedef int (*CompareOperator)(const T &, const T &);

  /**
   * match function
   * if match return true
   * if not match return false
   */
  typedef bool (*MatchOperator)(const T &);

  /**
   * modify function
   */
  typedef void (*ModifyOperator)(T &);

  SafeOrderList(CompareOperator operation) : mutex()
  {
    condition = operation;
  }

  bool empty()
  {
    TGuard gurad(mutex);
    return container.empty();
  }

  size_t size()
  {
    TGuard guard(mutex);
    return container.size();
  }

  void insert(const T &value)
  {
    TGuard guard(mutex);
    Iterator iter = container.begin();
    for (; iter != container.end(); iter++)
    {
      if (condition(value, *iter) > 0)
        break;
    }
    container.insert(iter, value);
  }

  void clear()
  {
    TGuard guard(mutex);
    container.clear();
  }

  T &front()
  {
    TGuard guard(mutex);
    return container.front();
  }

  T &back()
  {
    TGuard guard(mutex);
    return container.front();
  }

  void pop_front()
  {
    TGuard guard(mutex);
    container.pop_front();
  }

  void pop_back()
  {
    TGuard guard(mutex);
    container.pop_back();
  }

  void remove_first_match(MatchOperator match)
  {
    TGuard guard(mutex);
    Iterator iter = container.begin();
    for (; iter != container.end(); iter++)
    {
      if (match(*iter))
      {
        container.erase(iter);
        break;
      }
    }
  }

  template <typename MatchFun>
  void remove_first_match(MatchFun match)
  {
    TGuard guard(mutex);
    Iterator iter = container.begin();
    for (; iter != container.end(); iter++)
    {
      if (match(*iter))
      {
        container.erase(iter);
        break;
      }
    }
  }

  void remove_all_match(MatchOperator match)
  {
    TGuard guard(mutex);
    Iterator iter = container.begin();
    for (; iter != container.end();)
    {
      if (match(*iter))
        container.erase(iter++);
      else
        iter++;
    }
  }

  template <typename MatchFun>
  void remove_all_match(MatchFun match)
  {
    TGuard guard(mutex);
    Iterator iter = container.begin();
    for (; iter != container.end();)
    {
      if (match(*iter))
      {
          Iterator willRemove = iter;
          iter++;
          container.erase(willRemove);
      }
      else
        iter++;
    }
  }

  void for_each(ModifyOperator operation)
  {
    TGuard guard(mutex);
    Iterator iter = container.begin();
    for (; iter != container.end(); iter++)
    {
      operation(*iter);
    }
  }

  template <typename ModifyFun>
  void for_each(ModifyFun operation)
  {
    TGuard guard(mutex);
    Iterator iter = container.begin();
    for (; iter != container.end(); iter++)
    {
      operation(*iter);
    }
  }

  void copy_to_list(std::list<T> &dest)
  {
    TGuard guard(mutex);
    dest.insert(dest.end(), container.begin(), container.end());
  }

private:
  TContainer container;
  CompareOperator condition;
  ncnn::Mutex mutex;
};

} // namespace ncnn

#endif
