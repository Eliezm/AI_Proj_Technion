(define (problem blocksworld-large-127)
  (:domain blocksworld)
  (:objects b0 b1 b10 b2 b3 b4 b5 b6 b7 b8 b9)
  (:init
    (on-table b2) (on-table b6) (on b0 b1) (on b1 b7) (on b10 b4) (on b3 b0) (on b4 b8) (on b5 b3) (on b7 b9) (on b8 b2) (on b9 b6) (clear b10) (clear b5) (arm-empty)
  )
  (:goal
    (and (on-table b0) (on b1 b0) (on b10 b9) (on b2 b1) (on b3 b2) (on b4 b3) (on b5 b4) (on b6 b5) (on b7 b6) (on b8 b7) (on b9 b8) (clear b10) (arm-empty))
  )
)
