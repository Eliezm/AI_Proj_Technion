(define (problem logistics-example)
  (:domain logistics-strips)
  (:objects
    pkg-0 pkg-1 - OBJ
    truck-0 - TRUCK
    airplane-0 - AIRPLANE
    loc-city-0-0 loc-city-0-1 loc-city-1-0 loc-city-1-1 - LOCATION
    city-0 city-1 - CITY
  )
  (:init
    (at airplane-0 loc-city-1-0) (at pkg-0 loc-city-0-1) (at pkg-1 loc-city-0-1) (at truck-0 loc-city-0-1) (in-city loc-city-0-0 city-0) (in-city loc-city-0-1 city-0) (in-city loc-city-1-0 city-1) (in-city loc-city-1-1 city-1) (AIRPORT loc-city-0-0) (AIRPORT loc-city-1-0) (AIRPORT loc-city-1-1)
  )
  (:goal
    (and (at pkg-0 loc-city-0-1) (at pkg-1 loc-city-0-1))
  )
)
