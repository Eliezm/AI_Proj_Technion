(define (problem logistics-3)
  (:domain logistics)
  (:objects
    truck-city0-0 truck-city1-0 - truck
    loc-city0-0 loc-city0-1 loc-city1-0 loc-city1-1 - location
    obj0 obj1 obj2 obj3 - object
    city0 city1 - city
  )
  (:init
    (at-obj obj0 loc-city0-1) (obj-at-city obj0 city0) (at-obj obj1 loc-city0-1) (obj-at-city obj1 city0) (at-obj obj2 loc-city0-0) (obj-at-city obj2 city0) (at-obj obj3 loc-city0-1) (obj-at-city obj3 city0) (at truck-city0-0 loc-city0-0) (truck-at-city truck-city0-0 city0) (at truck-city1-0 loc-city1-0) (truck-at-city truck-city1-0 city1) (connected loc-city0-0 loc-city0-1) (connected loc-city0-1 loc-city0-0) (connected loc-city1-0 loc-city1-1) (connected loc-city1-1 loc-city1-0) (connected loc-city0-0 loc-city1-0) (connected loc-city1-0 loc-city0-0)
  )
  (:goal (and
    (at-obj obj0 loc-city0-0) (at-obj obj1 loc-city0-0) (at-obj obj2 loc-city0-0) (at-obj obj3 loc-city0-0)
  ))
)
