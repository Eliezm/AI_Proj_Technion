(define (problem blocksworld-large-75)
  (:domain blocksworld)
  (:objects b0 b1 b10 b2 b3 b4 b5 b6 b7 b8 b9)
  (:init
    (on-table b2) (on-table b3) (on-table b6) (on b0 b3) (on b1 b6) (on b10 b1) (on b4 b5) (on b5 b0) (on b7 b2) (on b8 b7) (on b9 b8) (clear b10) (clear b4) (clear b9) (arm-empty)
  )
  (:goal
    (and (on-table b4) (on-table b7) (on-table b8) (on b0 b5) (on b1 b10) (on b10 b9) (on b2 b7) (on b3 b1) (on b5 b3) (on b6 b8) (on b9 b6) (clear b0) (clear b2) (clear b4) (arm-empty))
  )
)
