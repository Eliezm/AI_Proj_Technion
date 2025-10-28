(define (problem blocksworld-large-7)
  (:domain blocksworld)
  (:objects b0 b1 b10 b11 b2 b3 b4 b5 b6 b7 b8 b9)
  (:init
    (on-table b10) (on-table b11) (on-table b3) (on b0 b3) (on b1 b10) (on b2 b4) (on b4 b8) (on b5 b0) (on b6 b2) (on b7 b5) (on b8 b9) (on b9 b7) (clear b1) (clear b11) (clear b6) (arm-empty)
  )
  (:goal
    (and (on-table b0) (on b1 b0) (on b10 b9) (on b11 b10) (on b2 b1) (on b3 b2) (on b4 b3) (on b5 b4) (on b6 b5) (on b7 b6) (on b8 b7) (on b9 b8) (clear b11) (arm-empty))
  )
)
