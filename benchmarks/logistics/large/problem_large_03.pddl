(define (problem logistics-3)
  (:domain logistics)
  (:objects
    truck-city0-0 truck-city0-1 truck-city1-0 truck-city1-1 truck-city2-0 truck-city2-1 truck-city3-0 truck-city3-1 - truck
    loc-city0-0 loc-city0-1 loc-city0-2 loc-city1-0 loc-city1-1 loc-city1-2 loc-city2-0 loc-city2-1 loc-city2-2 loc-city3-0 loc-city3-1 loc-city3-2 - location
    obj0 obj1 obj2 obj3 obj4 obj5 obj6 obj7 obj8 obj9 obj10 obj11 - object
    city0 city1 city2 city3 - city
  )
  (:init
    (at-obj obj0 loc-city1-1) (obj-at-city obj0 city1) (at-obj obj1 loc-city0-1) (obj-at-city obj1 city0) (at-obj obj2 loc-city1-0) (obj-at-city obj2 city1) (at-obj obj3 loc-city3-0) (obj-at-city obj3 city3) (at-obj obj4 loc-city3-2) (obj-at-city obj4 city3) (at-obj obj5 loc-city1-2) (obj-at-city obj5 city1) (at-obj obj6 loc-city1-0) (obj-at-city obj6 city1) (at-obj obj7 loc-city3-1) (obj-at-city obj7 city3) (at-obj obj8 loc-city2-1) (obj-at-city obj8 city2) (at-obj obj9 loc-city2-0) (obj-at-city obj9 city2) (at-obj obj10 loc-city3-1) (obj-at-city obj10 city3) (at-obj obj11 loc-city2-1) (obj-at-city obj11 city2) (at truck-city0-0 loc-city0-0) (truck-at-city truck-city0-0 city0) (at truck-city0-1 loc-city0-0) (truck-at-city truck-city0-1 city0) (at truck-city1-0 loc-city1-0) (truck-at-city truck-city1-0 city1) (at truck-city1-1 loc-city1-0) (truck-at-city truck-city1-1 city1) (at truck-city2-0 loc-city2-0) (truck-at-city truck-city2-0 city2) (at truck-city2-1 loc-city2-0) (truck-at-city truck-city2-1 city2) (at truck-city3-0 loc-city3-0) (truck-at-city truck-city3-0 city3) (at truck-city3-1 loc-city3-0) (truck-at-city truck-city3-1 city3) (connected loc-city0-0 loc-city0-1) (connected loc-city0-1 loc-city0-0) (connected loc-city0-1 loc-city0-2) (connected loc-city0-2 loc-city0-1) (connected loc-city1-0 loc-city1-1) (connected loc-city1-1 loc-city1-0) (connected loc-city1-1 loc-city1-2) (connected loc-city1-2 loc-city1-1) (connected loc-city2-0 loc-city2-1) (connected loc-city2-1 loc-city2-0) (connected loc-city2-1 loc-city2-2) (connected loc-city2-2 loc-city2-1) (connected loc-city3-0 loc-city3-1) (connected loc-city3-1 loc-city3-0) (connected loc-city3-1 loc-city3-2) (connected loc-city3-2 loc-city3-1) (connected loc-city0-0 loc-city1-0) (connected loc-city1-0 loc-city0-0) (connected loc-city1-0 loc-city2-0) (connected loc-city2-0 loc-city1-0) (connected loc-city2-0 loc-city3-0) (connected loc-city3-0 loc-city2-0)
  )
  (:goal (and
    (at-obj obj0 loc-city0-0) (at-obj obj1 loc-city0-0) (at-obj obj2 loc-city0-0) (at-obj obj3 loc-city0-0) (at-obj obj4 loc-city0-0) (at-obj obj5 loc-city0-0) (at-obj obj6 loc-city0-0) (at-obj obj7 loc-city0-0) (at-obj obj8 loc-city0-0) (at-obj obj9 loc-city0-0) (at-obj obj10 loc-city0-0) (at-obj obj11 loc-city0-0)
  ))
)
