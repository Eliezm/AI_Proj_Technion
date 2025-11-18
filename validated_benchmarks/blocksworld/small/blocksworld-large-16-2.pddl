(define (problem blocksworld-large-16)
  (:domain blocksworld)
  (:objects b0 b1 b10 b2 b3 b4 b5 b6 b7 b8 b9)
  (:init
    (on-table b0) (on-table b3) (on-table b4) (on-table b7) (on b1 b0) (on b10 b8) (on b2 b10) (on b5 b3) (on b6 b1) (on b8 b6) (on b9 b5) (clear b2) (clear b4) (clear b7) (clear b9) (arm-empty)
  )
  (:goal
    (and (on-table b0) (on b1 b0) (on b10 b9) (on b2 b1) (on b3 b2) (on b4 b3) (on b5 b4) (on b6 b5) (on b7 b6) (on b8 b7) (on b9 b8) (clear b10) (arm-empty))
  )
)
