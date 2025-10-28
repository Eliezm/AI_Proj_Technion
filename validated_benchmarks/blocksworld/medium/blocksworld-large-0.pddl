(define (problem blocksworld-large-0)
  (:domain blocksworld)
  (:objects b0 b1 b10 b11 b2 b3 b4 b5 b6 b7 b8 b9)
  (:init
    (on-table b0) (on-table b4) (on-table b5) (on-table b7) (on-table b9) (on b1 b8) (on b10 b9) (on b11 b5) (on b2 b1) (on b3 b7) (on b6 b4) (on b8 b6) (clear b0) (clear b10) (clear b11) (clear b2) (clear b3) (arm-empty)
  )
  (:goal
    (and (on-table b0) (on b1 b0) (on b10 b9) (on b11 b10) (on b2 b1) (on b3 b2) (on b4 b3) (on b5 b4) (on b6 b5) (on b7 b6) (on b8 b7) (on b9 b8) (clear b11) (arm-empty))
  )
)
