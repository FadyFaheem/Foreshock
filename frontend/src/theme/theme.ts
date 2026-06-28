import { createTheme } from '@mui/material/styles';

// Foreshock brand. Adjust these to rebrand; components reference the theme
// (never hardcoded colors) so this file is the single source of truth.
export const BRAND_COLORS = {
  primary: '#1f6feb',
  primaryDark: '#1452b3',
  primaryLight: '#4f8ef5',
  secondary: '#f59e0b',
  secondaryDark: '#b9770a',
  secondaryLight: '#ffb733',
  white: '#FFFFFF',
  lightGray: '#F5F7FA',
  mediumGray: '#6b7280',
  darkGray: '#1f2430',
} as const;

// Per-condition colors, used by the prediction bars and (optionally) charts.
export const CONDITION_COLORS: Record<string, string> = {
  normal: '#2e7d32',
  inner_race: '#ed6c02',
  outer_race: '#d32f2f',
  ball: '#7b1fa2',
  // IMS run-to-failure phases (v2 health embedding).
  healthy: '#2e7d32',
  degrading: '#d32f2f',
};

export const chartSeriesColors = [
  BRAND_COLORS.primary,
  BRAND_COLORS.secondary,
  BRAND_COLORS.darkGray,
  BRAND_COLORS.mediumGray,
] as const;

const BASE_THEME = createTheme();

export const appTheme = createTheme({
  palette: {
    primary: {
      main: BRAND_COLORS.primary,
      dark: BRAND_COLORS.primaryDark,
      light: BRAND_COLORS.primaryLight,
      contrastText: BRAND_COLORS.white,
    },
    secondary: {
      main: BRAND_COLORS.secondary,
      dark: BRAND_COLORS.secondaryDark,
      light: BRAND_COLORS.secondaryLight,
      contrastText: BRAND_COLORS.white,
    },
    text: {
      primary: BRAND_COLORS.darkGray,
      secondary: BRAND_COLORS.mediumGray,
    },
    background: {
      default: BRAND_COLORS.lightGray,
      paper: BRAND_COLORS.white,
    },
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          color: BRAND_COLORS.darkGray,
          backgroundColor: BRAND_COLORS.lightGray,
        },
      },
    },
    // Full-screen dialogs on mobile -- matches mobile-first UX patterns.
    MuiDialog: {
      styleOverrides: {
        root: {
          [BASE_THEME.breakpoints.down('sm')]: {
            '& .MuiDialog-paper': {
              margin: 0,
              width: '100%',
              maxWidth: '100%',
              maxHeight: '100%',
              height: '100%',
              borderRadius: 0,
            },
          },
        },
      },
    },
  },
});
