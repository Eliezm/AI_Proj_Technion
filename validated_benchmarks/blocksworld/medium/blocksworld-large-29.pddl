(define (problem blocksworld-large-29)
  (:domain blocksworld)
  (:objects b0 b1 b10 b11 b12 b2 b3 b4 b5 b6 b7 b8 b9)
  (:init
    (on-table b0) (on-table b11) (on-table b6) (on-table b8) (on b1 b2) (on b10 b9) (on b12 b10) (on b2 b12) (on b3 b5) (on b5 b0) (on b7 b1) (on b9 b11) (clear b3) (clear b6) (clear b7) (clear b8) (holding b4)
  )
  (:goal
    (and (on-table b0) (on b1 b0) (on b10 b9) (on b11 b10) (on b12 b11) (on b2 b1) (on b3 b2) (on b4 b3) (on b5 b4) (on b6 b5) (on b7 b6) (on b8 b7) (on b9 b8) (clear b12) (arm-empty))
  )
)
