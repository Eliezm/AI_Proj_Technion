(define (problem logistics-0)
  (:domain logistics)
  (:objects
    truck-city0-0 truck-city1-0 truck-city2-0 - truck
    loc-city0-0 loc-city0-1 loc-city1-0 loc-city1-1 loc-city2-0 loc-city2-1 - location
    obj0 obj1 obj2 obj3 obj4 obj5 obj6 obj7 - object
    city0 city1 city2 - city
  )
  (:init
    (at-obj obj0 loc-city0-0) (obj-at-city obj0 city0) (at-obj obj1 loc-city0-1) (obj-at-city obj1 city0) (at-obj obj2 loc-city2-1) (obj-at-city obj2 city2) (at-obj obj3 loc-city1-1) (obj-at-city obj3 city1) (at-obj obj4 loc-city1-0) (obj-at-city obj4 city1) (at-obj obj5 loc-city1-0) (obj-at-city obj5 city1) (at-obj obj6 loc-city0-1) (obj-at-city obj6 city0) (at-obj obj7 loc-city0-1) (obj-at-city obj7 city0) (at truck-city0-0 loc-city0-0) (truck-at-city truck-city0-0 city0) (at truck-city1-0 loc-city1-0) (truck-at-city truck-city1-0 city1) (at truck-city2-0 loc-city2-0) (truck-at-city truck-city2-0 city2) (connected loc-city0-0 loc-city0-1) (connected loc-city0-1 loc-city0-0) (connected loc-city1-0 loc-city1-1) (connected loc-city1-1 loc-city1-0) (connected loc-city2-0 loc-city2-1) (connected loc-city2-1 loc-city2-0) (connected loc-city0-0 loc-city1-0) (connected loc-city1-0 loc-city0-0) (connected loc-city1-0 loc-city2-0) (connected loc-city2-0 loc-city1-0)
  )
  (:goal (and
    (at-obj obj0 loc-city0-0) (at-obj obj1 loc-city0-0) (at-obj obj2 loc-city0-0) (at-obj obj3 loc-city0-0) (at-obj obj4 loc-city0-0) (at-obj obj5 loc-city0-0) (at-obj obj6 loc-city0-0) (at-obj obj7 loc-city0-0)
  ))
)
