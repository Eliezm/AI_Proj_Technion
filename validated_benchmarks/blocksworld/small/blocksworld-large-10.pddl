(define (problem blocksworld-large-10)
  (:domain blocksworld)
  (:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9)
  (:init
    (on-table b0) (on-table b1) (on b2 b0) (on b3 b2) (on b4 b8) (on b5 b6) (on b6 b9) (on b7 b5) (on b8 b3) (on b9 b4) (clear b1) (clear b7) (arm-empty)
  )
  (:goal
    (and (on-table b0) (on b1 b0) (on b2 b1) (on b3 b2) (on b4 b3) (on b5 b4) (on b6 b5) (on b7 b6) (on b8 b7) (on b9 b8) (clear b9) (arm-empty))
  )
)
