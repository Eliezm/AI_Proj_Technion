(define (problem blocksworld-large-61)
  (:domain blocksworld)
  (:objects b0 b1 b10 b2 b3 b4 b5 b6 b7 b8 b9)
  (:init
    (on-table b10) (on-table b3) (on-table b5) (on b0 b10) (on b1 b3) (on b2 b1) (on b4 b2) (on b6 b9) (on b7 b6) (on b8 b5) (on b9 b8) (clear b0) (clear b4) (clear b7) (arm-empty)
  )
  (:goal
    (and (on-table b0) (on b1 b0) (on b10 b9) (on b2 b1) (on b3 b2) (on b4 b3) (on b5 b4) (on b6 b5) (on b7 b6) (on b8 b7) (on b9 b8) (clear b10) (arm-empty))
  )
)
